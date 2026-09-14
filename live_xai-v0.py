"""Live SHAP + RW-UCFI inference for the production model.

The implementation mirrors Cell 09 (Kernel SHAP) and Cell 10 (RW-UCFI)
from Semhas_Final_Pipeline_Locked_SE_0809_FIXED(2).ipynb, but applies them
post-hoc to new-customer production predictions.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from inference_engine import prepare_new_data


DEFAULT_RISK_WEIGHTS = {"Low Risk": 0.50, "Medium Risk": 1.00, "High Risk": 1.50}
DEFAULT_ALPHA = 1.00
DEFAULT_UNCERTAIN_FLAG_BONUS = 0.25


def _normalize_shap_values(raw_values, n_samples: int, n_features: int, n_classes: int) -> np.ndarray:
    """Normalize SHAP multiclass formats to samples x features x classes."""
    if isinstance(raw_values, list):
        arr = np.stack(raw_values, axis=2)
    else:
        arr = np.asarray(raw_values)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        elif arr.ndim == 3:
            if arr.shape[0] == n_classes and arr.shape[1] == n_samples:
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.shape[0] == n_samples and arr.shape[1] == n_features:
                pass
            else:
                raise ValueError(f"Unrecognized SHAP array shape: {arr.shape}")
        else:
            raise ValueError(f"Unsupported SHAP value dimension: {arr.ndim}")

    if arr.shape[0] != n_samples:
        raise ValueError(f"SHAP sample dimension mismatch: {arr.shape[0]} vs {n_samples}")
    if arr.shape[1] != n_features:
        raise ValueError(f"SHAP feature dimension mismatch: {arr.shape[1]} vs {n_features}")
    if arr.shape[2] != n_classes:
        # This app is designed for the 3-class production classifier.
        raise ValueError(f"SHAP class dimension mismatch: {arr.shape[2]} vs {n_classes}")
    return arr


def _risk_weight_for_label(label: str, weights: dict[str, float]) -> float:
    low = str(label).lower()
    if "high" in low:
        return float(weights.get("High Risk", 1.50))
    if "medium" in low or "moderate" in low:
        return float(weights.get("Medium Risk", 1.00))
    if "low" in low:
        return float(weights.get("Low Risk", 0.50))
    return 1.00


def load_background_artifact(path: str | os.PathLike) -> pd.DataFrame:
    obj = joblib.load(path)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    if isinstance(obj, dict) and isinstance(obj.get("X_background"), pd.DataFrame):
        return obj["X_background"].copy()
    raise ValueError("Background artifact must contain a DataFrame or {'X_background': DataFrame}.")


def build_background_from_raw(reference_raw: pd.DataFrame, bundle: dict, background_size: int, random_state: int = 42) -> pd.DataFrame:
    X_ref, _ = prepare_new_data(reference_raw, bundle)
    if len(X_ref) == 0:
        raise ValueError("Reference/background dataset is empty.")
    n = min(int(background_size), len(X_ref))
    return X_ref.sample(n=n, random_state=int(random_state)).reset_index(drop=True)


def compute_live_xai(
    explain_raw: pd.DataFrame,
    prediction_output: pd.DataFrame,
    bundle: dict,
    *,
    background_X: pd.DataFrame | None = None,
    background_raw: pd.DataFrame | None = None,
    explain_rows: int = 3,
    background_size: int = 30,
    top_n: int = 10,
    nsamples: int = 200,
    risk_weights: dict[str, float] | None = None,
    uncertainty_alpha: float = DEFAULT_ALPHA,
    uncertain_flag_bonus: float = DEFAULT_UNCERTAIN_FLAG_BONUS,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run Kernel SHAP and RW-UCFI on selected new-customer rows.

    Global live importance is aggregated over the rows explained in this run,
    using the same formulas as the dissertation pipeline:
      SHAP global = mean(abs(SHAP)) across samples and classes.
      RW-UCFI = abs(SHAP) * class risk weight * uncertainty multiplier.
      RW-UCFI global = mean(RW-UCFI tensor) across samples and classes.
    """
    try:
        import shap
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Package 'shap' is not installed. Run pip install -r requirements.txt.") from exc

    model = bundle["model"]
    encoder = bundle["label_encoder"]
    feature_names = list(bundle["selected_features"])
    class_names = [str(c) for c in bundle.get("classes", list(encoder.classes_))]
    clean_to_original = bundle.get("clean_to_original", {}) or {}

    n_explain = min(max(1, int(explain_rows)), len(explain_raw))
    explain_raw_sel = explain_raw.iloc[:n_explain].copy()
    pred_sel = prediction_output.iloc[:n_explain].reset_index(drop=True).copy()
    X_explain, _ = prepare_new_data(explain_raw_sel, bundle)
    X_explain = X_explain[feature_names].reset_index(drop=True)

    if background_X is None:
        if background_raw is None:
            raise ValueError(
                "Live SHAP requires a reference/background dataset. Provide production_shap_background.joblib "
                "or point the app to the pipeline raw dataset."
            )
        background_X = build_background_from_raw(background_raw, bundle, background_size, random_state)
    else:
        background_X = background_X.copy()
        missing = [c for c in feature_names if c not in background_X.columns]
        if missing:
            raise ValueError(f"Background artifact is missing {len(missing)} selected features; sample: {missing[:5]}")
        background_X = background_X[feature_names]
        if len(background_X) > int(background_size):
            background_X = background_X.sample(n=int(background_size), random_state=int(random_state))
        background_X = background_X.reset_index(drop=True)

    if len(background_X) < 2:
        raise ValueError("SHAP background must contain at least 2 rows.")

    def predict_proba_wrapper(input_data):
        if isinstance(input_data, pd.DataFrame):
            X_input = input_data.copy()
        else:
            X_input = pd.DataFrame(input_data, columns=feature_names)
        return model.predict_proba(X_input[feature_names])

    explainer = shap.KernelExplainer(predict_proba_wrapper, background_X)
    raw_values = explainer.shap_values(X_explain, nsamples=int(nsamples))
    shap_values = _normalize_shap_values(
        raw_values,
        n_samples=len(X_explain),
        n_features=len(feature_names),
        n_classes=len(class_names),
    )

    # Prediction/uncertainty alignment.
    pred_encoded = pred_sel["predicted_riskLevel_encoded"].astype(int).to_numpy()
    pred_labels = pred_sel["predicted_riskLevel"].astype(str).to_numpy()
    max_proba = pd.to_numeric(pred_sel["max_proba"], errors="coerce").fillna(0.0).to_numpy(float)
    margin = pd.to_numeric(pred_sel["margin"], errors="coerce").fillna(0.0).to_numpy(float)
    entropy = pd.to_numeric(pred_sel["entropy_norm"], errors="coerce").fillna(0.0).to_numpy(float)
    uncertain = pred_sel["is_uncertain"].astype(bool).to_numpy()

    uncertainty_score = ((1.0 - max_proba) + (1.0 - margin) + entropy) / 3.0
    uncertainty_multiplier = 1.0 + float(uncertainty_alpha) * uncertainty_score + float(uncertain_flag_bonus) * uncertain.astype(int)

    weights = dict(DEFAULT_RISK_WEIGHTS)
    if risk_weights:
        weights.update({k: float(v) for k, v in risk_weights.items()})
    class_weights = np.array([_risk_weight_for_label(c, weights) for c in class_names], dtype=float)

    abs_shap = np.abs(shap_values)
    rw_tensor = (
        abs_shap
        * class_weights.reshape(1, 1, -1)
        * uncertainty_multiplier.reshape(-1, 1, 1)
    )

    # Local outputs: predicted class for each entity, matching Cell 09.12 / Cell 10.11 logic.
    local_shap_rows: list[dict[str, Any]] = []
    local_rw_rows: list[dict[str, Any]] = []
    for sample_pos in range(len(X_explain)):
        class_idx = int(pred_encoded[sample_pos])
        if class_idx >= shap_values.shape[2]:
            class_idx = shap_values.shape[2] - 1
        shap_vec = shap_values[sample_pos, :, class_idx]
        rw_vec = rw_tensor[sample_pos, :, class_idx]
        xrow = X_explain.iloc[sample_pos]

        shap_order = np.argsort(np.abs(shap_vec))[::-1]
        rw_order = np.argsort(rw_vec)[::-1]
        shap_rank = {int(pos): rank + 1 for rank, pos in enumerate(shap_order)}
        rw_rank = {int(pos): rank + 1 for rank, pos in enumerate(rw_order)}

        for feature_pos, feature in enumerate(feature_names):
            common = {
                "sample_position": int(sample_pos),
                "input_row": int(sample_pos + 1),
                "predicted_class": str(pred_labels[sample_pos]),
                "predicted_class_index": int(pred_encoded[sample_pos]),
                "feature": feature,
                "original_column": clean_to_original.get(feature, feature),
                "feature_value": xrow.get(feature),
            }
            local_shap_rows.append({
                **common,
                "shap_value": float(shap_vec[feature_pos]),
                "shap_abs": float(abs(shap_vec[feature_pos])),
                "local_shap_rank": int(shap_rank[feature_pos]),
            })
            local_rw_rows.append({
                **common,
                "shap_abs_predicted_class": float(abs_shap[sample_pos, feature_pos, class_idx]),
                "risk_weight_predicted_class": float(class_weights[class_idx]),
                "uncertainty_score": float(uncertainty_score[sample_pos]),
                "uncertainty_multiplier": float(uncertainty_multiplier[sample_pos]),
                "rw_ucfi_score_entity": float(rw_vec[feature_pos]),
                "local_rw_ucfi_rank": int(rw_rank[feature_pos]),
            })

    local_shap = pd.DataFrame(local_shap_rows)
    local_rw = pd.DataFrame(local_rw_rows)

    # Global outputs over explained live rows, matching pipeline formulas.
    global_shap_score = abs_shap.mean(axis=(0, 2))
    global_rw_score = rw_tensor.mean(axis=(0, 2))
    global_shap = pd.DataFrame({
        "feature": feature_names,
        "original_column": [clean_to_original.get(c, c) for c in feature_names],
        "mean_abs_shap_global": global_shap_score,
    }).sort_values("mean_abs_shap_global", ascending=False).reset_index(drop=True)
    global_shap["shap_global_rank"] = np.arange(1, len(global_shap) + 1)

    global_rw = pd.DataFrame({
        "feature": feature_names,
        "original_column": [clean_to_original.get(c, c) for c in feature_names],
        "rw_ucfi_score_global": global_rw_score,
    }).sort_values("rw_ucfi_score_global", ascending=False).reset_index(drop=True)
    global_rw["rw_ucfi_global_rank"] = np.arange(1, len(global_rw) + 1)

    sample_summary = pd.DataFrame({
        "sample_position": np.arange(len(X_explain), dtype=int),
        "input_row": np.arange(1, len(X_explain) + 1, dtype=int),
        "predicted_class": pred_labels,
        "predicted_class_index": pred_encoded,
        "max_proba": max_proba,
        "margin": margin,
        "entropy_norm": entropy,
        "is_uncertain": uncertain,
        "uncertainty_score": uncertainty_score,
        "uncertainty_multiplier": uncertainty_multiplier,
    })

    return {
        "local_shap": local_shap,
        "local_rw_ucfi": local_rw,
        "global_shap": global_shap,
        "global_rw_ucfi": global_rw,
        "sample_summary": sample_summary,
        "metadata": {
            "explainer_type": "KernelExplainer_model_agnostic",
            "n_explained_rows": int(len(X_explain)),
            "background_rows": int(len(background_X)),
            "n_features": int(len(feature_names)),
            "classes": class_names,
            "nsamples": int(nsamples),
            "top_n": int(top_n),
            "risk_weight_map": {c: float(_risk_weight_for_label(c, weights)) for c in class_names},
            "uncertainty_alpha": float(uncertainty_alpha),
            "uncertain_flag_bonus": float(uncertain_flag_bonus),
            "global_scope": "rows explained in this live inference run",
        },
    }


def xai_results_to_excel_bytes(results: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for key, sheet in [
            ("sample_summary", "Summary"),
            ("local_shap", "Local_SHAP"),
            ("local_rw_ucfi", "Local_RW_UCFI"),
            ("global_shap", "Global_SHAP"),
            ("global_rw_ucfi", "Global_RW_UCFI"),
        ]:
            df = results.get(key)
            if isinstance(df, pd.DataFrame):
                df.to_excel(writer, sheet_name=sheet, index=False)
        pd.DataFrame([results.get("metadata", {})]).to_excel(writer, sheet_name="Metadata", index=False)
    return buffer.getvalue()
