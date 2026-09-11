"""Inference helpers ported from Semhas_Final_Pipeline_Locked_SE_0809_FIXED."""
from __future__ import annotations

import io
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model_compat import WeightedOOFStackingClassifier

# The notebook saved the custom estimator while executed as __main__.
# Registering it here makes joblib loading robust when Streamlit runs app.py.
setattr(sys.modules.get("__main__"), "WeightedOOFStackingClassifier", WeightedOOFStackingClassifier)

DEFAULT_THRESHOLDS = {
    "max_proba_threshold": 0.60,
    "margin_threshold": 0.15,
    "entropy_norm_threshold": 0.80,
}


def sanitize_name(name):
    name = str(name).strip()
    name = name.replace("%", "pct")
    name = name.replace("&", "and")
    name = re.sub(r"[^\w]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    if name == "":
        name = "unnamed_column"
    if name[0].isdigit():
        name = "col_" + name
    return name


def make_unique(names):
    seen = {}
    unique_names = []
    for name in names:
        if name not in seen:
            seen[name] = 0
            unique_names.append(name)
        else:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
    return unique_names


def clean_columns(df):
    original_cols = list(df.columns)
    cleaned_cols = make_unique([sanitize_name(c) for c in original_cols])
    df_clean = df.copy()
    df_clean.columns = cleaned_cols
    colmap_df = pd.DataFrame({"original_column": original_cols, "clean_column": cleaned_cols})
    original_to_clean = dict(zip(original_cols, cleaned_cols))
    clean_to_original = dict(zip(cleaned_cols, original_cols))
    return df_clean, colmap_df, original_to_clean, clean_to_original


def is_date_like_column(col_name):
    col = str(col_name).lower()
    keywords = [
        "date", "tanggal", "tgl", "time", "period", "registered", "registerd",
        "created", "updated", "start", "end", "expired", "expiry",
    ]
    return any(k in col for k in keywords)


def convert_date_like_columns(df, min_parse_ratio=0.50):
    df_out = df.copy()
    for col in df_out.columns:
        if not is_date_like_column(col) or pd.api.types.is_numeric_dtype(df_out[col]):
            continue
        parsed = pd.to_datetime(df_out[col], errors="coerce")
        if parsed.notna().mean() >= min_parse_ratio:
            df_out[col] = parsed.map(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    return df_out


def calculate_uncertainty_from_proba(proba):
    proba = np.asarray(proba)
    max_proba = np.max(proba, axis=1)
    sorted_proba = np.sort(proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    eps = 1e-12
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    entropy_norm = entropy / np.log(proba.shape[1])
    return pd.DataFrame({
        "max_proba": max_proba,
        "margin": margin,
        "entropy_norm": entropy_norm,
    })


def assign_uncertainty_flag(uncertainty_df, max_proba_threshold=0.60, margin_threshold=0.15, entropy_norm_threshold=0.80):
    df = uncertainty_df.copy()
    df["is_uncertain_max_proba"] = df["max_proba"] < max_proba_threshold
    df["is_uncertain_margin"] = df["margin"] < margin_threshold
    df["is_uncertain_entropy"] = df["entropy_norm"] > entropy_norm_threshold
    df["is_uncertain"] = (
        df["is_uncertain_max_proba"]
        | df["is_uncertain_margin"]
        | df["is_uncertain_entropy"]
    )
    return df


def load_production_bundle(bundle_path):
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Production bundle not found: {bundle_path}")
    bundle = joblib.load(bundle_path)
    required = ["model", "label_encoder", "selected_features", "classes", "best_model_name", "pipeline"]
    missing = [k for k in required if k not in bundle]
    if missing:
        raise ValueError(f"Invalid production bundle. Missing keys: {missing}")
    return bundle


def build_inference_schema_report(df_new_raw, bundle):
    selected_features = bundle["selected_features"]
    df_new_clean, _, _, _ = clean_columns(df_new_raw)
    df_new_clean = convert_date_like_columns(df_new_clean)
    df_new_clean = df_new_clean.replace([np.inf, -np.inf], np.nan)

    missing_selected = [c for c in selected_features if c not in df_new_clean.columns]
    available = [c for c in selected_features if c in df_new_clean.columns]
    extra = [c for c in df_new_clean.columns if c not in selected_features]
    missing_ratio = {c: float(df_new_clean[c].isna().mean()) for c in available}
    return {
        "n_input_rows": int(df_new_raw.shape[0]),
        "n_input_columns_raw": int(df_new_raw.shape[1]),
        "n_input_columns_clean": int(df_new_clean.shape[1]),
        "n_selected_features_expected": int(len(selected_features)),
        "n_selected_features_available": int(len(available)),
        "n_selected_features_missing": int(len(missing_selected)),
        "missing_selected_features": missing_selected,
        "n_extra_clean_columns": int(len(extra)),
        "extra_clean_columns_sample": extra[:50],
        "selected_feature_missing_ratio_sample": dict(list(missing_ratio.items())[:50]),
        "schema_status": "PASS" if not missing_selected else "PASS_WITH_MISSING_FEATURES_FILLED_AS_NAN",
        "note": "Missing selected features are filled with NaN; trained preprocessing handles imputation.",
    }


def prepare_new_data(df_new_raw, bundle):
    selected_features = bundle["selected_features"]
    df_new_clean, colmap, _, _ = clean_columns(df_new_raw)
    df_new_clean = convert_date_like_columns(df_new_clean)
    df_new_clean = df_new_clean.replace([np.inf, -np.inf], np.nan)
    missing = [c for c in selected_features if c not in df_new_clean.columns]
    for col in missing:
        df_new_clean[col] = np.nan
    X_new = df_new_clean[selected_features].copy()
    report = build_inference_schema_report(df_new_raw, bundle)
    report.update({"new_column_mapping": colmap.to_dict(orient="records")[:100], "feature_order_enforced": True})
    return X_new, report


def predict_from_dataframe(df_new_raw, bundle_or_path):
    bundle = load_production_bundle(bundle_or_path) if isinstance(bundle_or_path, (str, os.PathLike, Path)) else bundle_or_path
    model = bundle["model"]
    encoder = bundle["label_encoder"]
    X_new, report = prepare_new_data(df_new_raw, bundle)

    pred_encoded = model.predict(X_new)
    pred_label = encoder.inverse_transform(np.asarray(pred_encoded, dtype=int))

    output = df_new_raw.copy()
    output["predicted_riskLevel"] = pred_label
    output["predicted_riskLevel_encoded"] = pred_encoded

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_new)
        for idx, cls in enumerate(encoder.classes_):
            output[f"proba_{cls}"] = proba[:, idx]
        u = calculate_uncertainty_from_proba(proba)
        thresholds = DEFAULT_THRESHOLDS | bundle.get("uncertainty_thresholds", {})
        u = assign_uncertainty_flag(
            u,
            max_proba_threshold=float(thresholds["max_proba_threshold"]),
            margin_threshold=float(thresholds["margin_threshold"]),
            entropy_norm_threshold=float(thresholds["entropy_norm_threshold"]),
        )
        output = pd.concat([output.reset_index(drop=True), u.reset_index(drop=True)], axis=1)

    report.update({
        "prediction_completed": True,
        "model_name": bundle.get("best_model_name"),
        "pipeline": bundle.get("pipeline"),
        "classes": list(bundle.get("classes", [])),
        "probability_available": bool(hasattr(model, "predict_proba")),
        "n_output_rows": int(output.shape[0]),
        "n_output_columns": int(output.shape[1]),
    })
    return output, report


def read_uploaded_table(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    raise ValueError("Only CSV, XLSX, and XLS files are supported.")


def build_input_template(bundle, n_rows=3):
    selected = list(bundle.get("selected_features", []))
    clean_to_original = bundle.get("clean_to_original", {}) or {}
    trace_cols = list(bundle.get("trace_cols", []))
    ordered_clean = []
    for c in trace_cols + selected:
        if c not in ordered_clean:
            ordered_clean.append(c)
    display_cols = [clean_to_original.get(c, c) for c in ordered_clean]
    return pd.DataFrame([{c: None for c in display_cols} for _ in range(n_rows)])


def dataframe_to_excel_bytes(df, sheet_name="Predictions"):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buffer.getvalue()


def json_bytes(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str).encode("utf-8")
