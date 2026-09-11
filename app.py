from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Required so joblib can restore the custom estimator saved from the notebook.
from model_compat import WeightedOOFStackingClassifier  # noqa: F401
from inference_engine import (
    build_input_template,
    dataframe_to_excel_bytes,
    load_production_bundle,
    predict_from_dataframe,
    read_uploaded_table,
)
from live_xai import (
    compute_live_xai,
    load_background_artifact,
    xai_results_to_excel_bytes,
)

st.set_page_config(
    page_title="B2B Customer Risk Profile",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent


def _first_existing_path(*candidates):
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            p = Path(candidate).expanduser()
            if p.exists():
                return p
        except Exception:
            pass
    return Path(candidates[-1]).expanduser()


# Dashboard code, pipeline output, and raw datasource do NOT need to live in the same folder.
# Windows research workspace is auto-detected when present; environment variables can override it.
DEFAULT_PIPELINE_ROOT = _first_existing_path(
    os.getenv("B2B_PIPELINE_ROOT"),
    Path(r"C:\S3\Semhas_New_Pipeline\output"),
    APP_DIR / "pipeline_output",
)
DEFAULT_DATA_ROOT = _first_existing_path(
    os.getenv("B2B_DATA_ROOT"),
    Path(r"C:\S3\Semhas_New_Pipeline\data"),
    APP_DIR / "data",
)

RISK_ORDER = ["Low Risk", "Medium Risk", "High Risk"]
RISK_SHORT = {"Low Risk": "Low", "Medium Risk": "Medium", "High Risk": "High"}
RISK_COLORS = {"Low Risk": "#477b5f", "Medium Risk": "#9b742f", "High Risk": "#aa4935"}
RISK_BG = {"Low Risk": "#e5efe9", "Medium Risk": "#f3ead7", "High Risk": "#f3e5df"}
INK = "#202522"
MUTED = "#777c77"
BORDER = "#d8d4ca"
CANVAS = "#f3f1ec"
PANEL = "#fbfaf7"
SIDEBAR = "#17211d"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Inter:wght@400;500;600&family=Libre+Caslon+Text:ital@0;1&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', Arial, sans-serif; color: {INK}; }}
.stApp {{ background: {CANVAS}; }}
.block-container {{ padding: 1.55rem 2.8rem 2.5rem 2.8rem; max-width: 1500px; }}
header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
[data-testid="stSidebar"] {{ background: {SIDEBAR}; border-right: 1px solid #26322d; min-width: 228px; max-width: 228px; }}
[data-testid="stSidebar"] > div:first-child {{ padding-top: 1.15rem; }}
[data-testid="stSidebar"] * {{ color: #e5e7e3; }}
[data-testid="stSidebar"] hr {{ border-color: #39413e; }}
[data-testid="stSidebar"] .stRadio label {{ padding: 0.55rem 0.25rem; font-size: .88rem; }}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child {{ display:none; }}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {{ gap: .25rem; }}
[data-testid="stSidebar"] .stRadio label:has(input:checked) {{ background:#2b3430; border-left:2px solid #d2a75d; border-radius:2px; padding-left:.65rem; }}
[data-testid="stSidebar"] .stRadio label:hover {{ background:#222d28; }}
.sidebar-brand {{ font-family:'Libre Caslon Text', Georgia, serif; font-style:italic; font-size:1.30rem; color:#fbfaf5; margin:.05rem 0 .55rem 0; }}
.sidebar-sub {{ font-size:0.814rem; line-height:1.45; color:#bfc5bf; letter-spacing:.01em; }}
.sidebar-bottom {{ position:fixed; left:1.8rem; bottom:1.65rem; width:175px; padding-top:1rem; border-top:1px solid #39413e; font-size:0.726rem; line-height:1.55; color:#86938c; }}
.sidebar-bottom b {{ color:#d8ded9; font-weight:500; }}
.page-title {{ font-family:'Libre Caslon Text', Georgia, serif; font-size:1.70rem; color:#222522; line-height:1.15; margin-bottom:.55rem; }}
.page-sub {{ color:#6f746f; font-size:0.880rem; margin-bottom:.15rem; }}
.top-meta {{ text-align:right; color:#8b8f89; font-size:0.770rem; line-height:1.55; }}
.top-meta .mono {{ font-family:'DM Mono', monospace; color:#626862; }}
.top-rule {{ border-top:1px solid #cbc7bd; margin:.9rem 0 1.65rem 0; }}
.panel {{ border:1px solid {BORDER}; background:{PANEL}; }}
.panel-title-row {{ display:flex; justify-content:space-between; align-items:baseline; padding:.88rem 1.25rem; border-bottom:1px solid {BORDER}; }}
.panel-title {{ font-family:'Libre Caslon Text', Georgia, serif; font-size:1.122rem; color:#222522; }}
.panel-note {{ font-size:0.715rem; color:#959891; }}
.kpi-grid {{ display:grid; grid-template-columns:repeat(4, 1fr); border:1px solid {BORDER}; background:{PANEL}; margin-bottom:1.12rem; }}
.kpi {{ padding:1.15rem 1.25rem 1.02rem 1.25rem; border-right:1px solid {BORDER}; min-height:104px; }}
.kpi:last-child {{ border-right:none; }}
.kpi-label {{ font-size:0.958rem; font-weight:700; color:#656a65; margin-bottom:.52rem; }}
.kpi-value {{ font-family:'Libre Caslon Text', Georgia, serif; font-size:1.925rem; line-height:1; color:#222522; }}
.kpi-value.high {{ color:{RISK_COLORS['High Risk']}; }} .kpi-value.medium {{ color:{RISK_COLORS['Medium Risk']}; }} .kpi-value.low {{ color:{RISK_COLORS['Low Risk']}; }}
.kpi-sub {{ font-size:0.737rem; color:#777b77; margin-top:.65rem; }}
.dist-wrap {{ padding:1.25rem 1.25rem 1.05rem 1.25rem; }}
.dist-row {{ display:grid; grid-template-columns:110px 1fr 58px; gap:.85rem; align-items:center; margin:.98rem 0; font-size:0.847rem; }}
.dist-track {{ height:13px; background:#e9e7e1; position:relative; }}
.dist-bar {{ height:13px; }}
.dist-pct {{ font-family:'DM Mono', monospace; font-size:0.770rem; color:#2e3430; text-align:right; }}
.sector-risk-wrap {{ padding:1.15rem 1.25rem 1.20rem 1.25rem; }}
.sector-risk-legend {{ display:flex; gap:1.25rem; align-items:center; justify-content:flex-end; margin-bottom:.85rem; font-size:0.704rem; color:#70756f; }}
.sector-risk-legend-item {{ display:flex; gap:.38rem; align-items:center; }}
.sector-risk-dot {{ width:9px; height:9px; display:inline-block; }}
.sector-risk-row {{ display:grid; grid-template-columns:minmax(155px, 220px) 1fr 205px; gap:.85rem; align-items:center; margin:.72rem 0; font-size:0.781rem; }}
.sector-risk-label {{ overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:#333834; }}
.sector-risk-track {{ height:18px; background:#e9e7e1; display:flex; overflow:hidden; }}
.sector-risk-seg {{ height:18px; min-width:0; }}
.sector-risk-values {{ font-family:'DM Mono', monospace; font-size:0.671rem; color:#666c67; text-align:right; white-space:nowrap; }}
@media (max-width: 950px) {{ .sector-risk-row {{ grid-template-columns:130px 1fr; }} .sector-risk-values {{ grid-column:2; text-align:left; }} }}
.table-wrap {{ overflow-x:auto; }}
.mock-table {{ width:100%; border-collapse:collapse; font-size:0.781rem; }}
.mock-table th {{ padding:.72rem .78rem; text-align:left; color:#858985; font-weight:500; border-bottom:1px solid #c9c5bc; font-family:'DM Mono', monospace; font-size:0.671rem; }}
.mock-table td {{ padding:.76rem .78rem; border-bottom:1px solid #dfdcd4; vertical-align:middle; }}
.mock-table tr:last-child td {{ border-bottom:none; }}
.risk-pill {{ display:inline-block; padding:.20rem .52rem; border-radius:2px; font-family:'DM Mono', monospace; font-size:0.660rem; }}
.mono {{ font-family:'DM Mono', monospace; }}
.search-count {{ color:#8d918b; font-size:0.726rem; }}
.customer-row {{ padding:.82rem 1rem; border-bottom:1px solid #dedbd3; background:{PANEL}; }}
.customer-row:last-child {{ border-bottom:0; }}
.customer-name {{ font-size:0.847rem; font-weight:600; color:#272c29; }}
.customer-sub {{ font-size:0.726rem; color:#959994; margin-top:.18rem; }}
.existing-list-head {{ display:flex; justify-content:space-between; align-items:baseline; gap:1rem; margin:.2rem 0 .55rem 0; }}
.existing-list-count {{ font-size:0.726rem; color:#8d918b; }}
.existing-card-title {{ font-size:0.847rem; font-weight:600; color:#272c29; line-height:1.35; }}
.existing-card-meta {{ font-size:0.704rem; color:#8d928c; margin-top:.20rem; line-height:1.45; }}
.existing-card-sector {{ font-size:0.726rem; color:#686e69; line-height:1.4; }}
.existing-selected {{ font-size:0.649rem; font-family:'DM Mono',monospace; color:#53677f; margin-top:.25rem; }}
.pagination-note {{ font-size:0.704rem; color:#8d918b; padding-top:.35rem; }}
.detail-head {{ padding:1.25rem 1.3rem 1.05rem 1.3rem; display:grid; grid-template-columns:1fr 120px; gap:1rem; }}
.detail-name {{ font-family:'Libre Caslon Text', Georgia, serif; font-size:1.430rem; margin-bottom:.55rem; }}
.detail-meta {{ color:#646a65; font-size:0.770rem; line-height:1.55; }}
.detail-risk {{ text-align:right; }}
.detail-risk-label {{ font-size:0.660rem; text-transform:uppercase; letter-spacing:.06em; color:#9a9c96; }}
.detail-risk-value {{ font-family:'Libre Caslon Text', Georgia, serif; font-style:italic; font-size:1.298rem; margin:.12rem 0 .28rem; }}
.detail-risk-prob {{ font-family:'DM Mono', monospace; font-size:0.737rem; color:#59605b; }}
.prediction-summary {{ overflow:hidden; }}
.prediction-confidence-section {{ padding:.82rem 1.3rem .92rem 1.3rem; border-top:1px solid {BORDER}; background:{PANEL}; }}
.confidence-wrap {{ width:100%; margin:0; min-width:0; max-width:none; box-sizing:border-box; }}
.confidence-row {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:.36rem; gap:.75rem; }}
.confidence-label {{ font-size:0.671rem; text-transform:uppercase; letter-spacing:.06em; color:#90958f; }}
.confidence-text {{ font-family:'DM Mono', monospace; font-size:0.715rem; color:#38557b; white-space:nowrap; }}
.confidence-track {{ width:100%; box-sizing:border-box; height:13px; background:linear-gradient(to right, #dcecff 0%, #a8c9f3 38%, #6d9ddd 68%, #285b9f 100%); border:1px solid #c9daee; position:relative; overflow:visible; }}
.confidence-marker {{ position:absolute; top:-5px; width:2px; height:21px; background:#1f2f43; transform:translateX(-1px); }}
.confidence-ticks {{ display:flex; justify-content:space-between; margin-top:.30rem; font-size:0.638rem; color:#8291a3; }}
.output-selector-note {{ font-size:0.715rem; color:#858a84; margin:-.15rem 0 .28rem 0; }}
.pred-chart {{ padding:1.08rem 1.3rem 1.25rem 1.3rem; border-top:1px solid {BORDER}; border-bottom:1px solid {BORDER}; }}
.pred-chart-title-row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.85rem; gap:1rem; }}
.pred-chart-title {{ font-family:'Libre Caslon Text', Georgia, serif; font-size:1.100rem; color:#242824; }}
.pred-chart-note {{ font-size:0.726rem; color:#8f948e; }}
.pred-chart-grid {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:1.4rem; align-items:end; min-height:250px; padding:.3rem 0 .15rem; }}
.pred-col {{ display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; }}
.pred-value {{ font-family:'DM Mono', monospace; font-size:0.726rem; color:#414744; margin-bottom:.35rem; }}
.pred-bar-wrap {{ width:100%; max-width:150px; height:190px; display:flex; align-items:flex-end; justify-content:center; }}
.pred-bar {{ width:100%; max-width:120px; min-height:8px; position:relative; }}
.pred-bar.hl {{ box-shadow:0 0 0 2px rgba(34,37,34,.08); }}
.pred-bar-label {{ margin-top:.55rem; font-size:0.792rem; color:#626863; text-align:center; }}
.pred-bar-label b {{ color:#252925; }}
.pred-badge {{ display:inline-block; margin-top:.35rem; padding:.16rem .45rem; background:#edf1f7; color:#53677f; border:1px solid #d4deeb; font-family:'DM Mono', monospace; font-size:0.638rem; }}
.explain-body {{ padding:1.0rem 1.25rem 1.15rem 1.25rem; }}
.driver-row {{ display:grid; grid-template-columns:minmax(165px, 1.8fr) 1fr 72px; gap:.75rem; align-items:center; margin:.60rem 0; font-size:0.748rem; }}
.driver-label {{ text-align:right; color:#646a66; }}
.driver-track {{ height:12px; display:flex; align-items:center; border-left:1px solid #d0ccc3; }}
.driver-pos {{ height:12px; background:#aa4935; min-width:2px; }}
.driver-neg {{ height:12px; background:#477b5f; min-width:2px; margin-left:auto; }}
.driver-value {{ font-family:'DM Mono', monospace; font-size:0.682rem; text-align:right; color:#505651; }}
.notice {{ border:1px solid {BORDER}; background:#f7f5f0; padding:.85rem 1rem; color:#777c77; font-size:0.748rem; line-height:1.55; }}
.path-card {{ padding:.95rem 1.2rem 1.15rem 1.2rem; }}
.path-code {{ font-family:'DM Mono', monospace; font-size:0.682rem; color:#92968f; }}
.path-name {{ font-size:0.836rem; font-weight:600; margin:.25rem 0; }}
.path-desc {{ color:#777c77; font-size:0.737rem; line-height:1.5; }}
.upload-note {{ color:#92958f; font-size:0.726rem; text-align:right; }}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {{ border-radius:0 !important; background:#fffefa !important; border-color:#cfcac0 !important; font-size:0.836rem !important; }}
.stButton button, .stDownloadButton button {{ border-radius:0 !important; border:1px solid #c9c4b9 !important; background:#fbfaf7 !important; color:#343a36 !important; font-size:0.825rem !important; }}
.stButton button[kind="primary"] {{ background:{SIDEBAR} !important; color:white !important; border-color:{SIDEBAR} !important; }}
[data-testid="stFileUploaderDropzone"] {{ border-radius:0; border:1px dashed #c6c1b7; background:#fdfcf9; }}
[data-testid="stMetric"] {{ background:#fbfaf7; border:1px solid {BORDER}; padding:.65rem .8rem; }}
.small {{ font-size:0.726rem; color:#858a84; }}
.warn-inline {{ font-size:0.715rem; color:#8f5d2d; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return html.escape(str(v))


def resolve_output_root(user_root: str | Path) -> Path:
    p = Path(user_root).expanduser()
    return p / "output" if (p / "output").exists() else p


def first_existing(*paths: Path):
    for p in paths:
        if p and p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_table(path_str: str):
    p = Path(path_str)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_excel(p)


@st.cache_data(show_spinner=False)
def load_json(path_str: str):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


@st.cache_resource(show_spinner=False)
def cached_bundle(path_str: str):
    return load_production_bundle(path_str)


@st.cache_data(show_spinner=False)
def cached_xai_background(path_str: str):
    return load_background_artifact(path_str)


@st.cache_data(show_spinner=False)
def load_rw_ucfi_tensor_artifact(path_str: str):
    """Load the Stage-4 compressed tensor used for exact filtered aggregation.

    The pipeline stores both ``abs_shap_values`` and ``rw_ucfi_tensor`` in the same
    artifact, together with feature names and the original test-set indices.
    Returning ordinary numpy arrays keeps downstream filtering deterministic.
    """
    path = Path(path_str)
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=True) as z:
            return {k: np.asarray(z[k]) for k in z.files}
    except Exception:
        return None


def artifact_paths(output_root: Path):
    s2 = output_root / "stage2_feature_filtering"
    s3 = output_root / "stage3_model_validation"
    s4 = output_root / "stage4_final_test_uncertainty"
    s5 = output_root / "stage5_production_inference"
    u = s4 / "uncertainty_analysis"
    shap = s4 / "explainability_shap"
    rw = s4 / "rw_ucfi_analysis"
    return {
        "feature_set": s2 / "final_feature_set.csv",
        "leaderboard": s3 / "validation_model_leaderboard.csv",
        "final_metrics": s4 / "best_model_final_test_metrics.json",
        "final_predictions": first_existing(u / "final_test_predictions_with_uncertainty.xlsx", u / "final_test_predictions_with_uncertainty.csv"),
        "shap_local": shap / "shap_local_explanation_predicted_class_top_features.csv",
        "shap_global": shap / "shap_global_importance.csv",
        "rw_entity": rw / "rw_ucfi_entity_level_top_features.csv",
        "rw_global": rw / "rw_ucfi_global_ranking.csv",
        "rw_tensor": rw / "rw_ucfi_tensor.npz",
        "rw_governance": rw / "rw_ucfi_governance_review_paths_entity_level.xlsx",
        "rw_summary": rw / "rw_ucfi_governance_review_path_summary.csv",
        "production_predictions": first_existing(s5 / "production_prediction_output.xlsx", s5 / "production_prediction_output.csv"),
        "production_metadata": s5 / "production_metadata.json",
        "production_summary": s5 / "production_uncertainty_summary.json",
        "bundle": s5 / "production_model_bundle.joblib",
        "xai_background": s5 / "production_shap_background.joblib",
        "selected_features": s5 / "production_selected_features.csv",
        "environment": output_root / "stage0_config" / "environment_info.json",
    }


def detect_col(df: pd.DataFrame | None, names):
    if df is None:
        return None
    direct = {str(c): c for c in df.columns}
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name in direct:
            return direct[name]
        if str(name).lower() in lower:
            return lower[str(name).lower()]
    return None


def normalize_bool(series):
    if series is None:
        return pd.Series(dtype=bool)
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def risk_class(raw) -> str:
    s = str(raw)
    low = s.lower()
    if "high" in low:
        return "High Risk"
    if "medium" in low or "moderate" in low:
        return "Medium Risk"
    if "low" in low:
        return "Low Risk"
    return s


def risk_pill(risk) -> str:
    r = risk_class(risk)
    return f'<span class="risk-pill" style="color:{RISK_COLORS.get(r, "#555")};background:{RISK_BG.get(r, "#eee")}">{esc(RISK_SHORT.get(r, r))}</span>'


def probability_col(df, risk):
    candidates = [f"proba_{risk}", f"P({risk})", f"p_{risk}"]
    return detect_col(df, candidates)


def to_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def fmt_prob(v):
    f = to_float(v)
    return "—" if np.isnan(f) else f"{f:.2f}"


def fmt_pct(v):
    f = to_float(v)
    return "—" if np.isnan(f) else f"{f * 100:.1f}%"


def company_name(row) -> str:
    for c in ["Company_Name", "Company Name", "Matched_Nama_CC", "Matched Nama CC", "longName", "shortName", "IDXCode", "symbol", "NIPNAS"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    idx = row.get("original_index", row.name)
    return f"Customer {idx}"


def customer_code(row) -> str:
    for c in ["NIPNAS", "IDXCode", "symbol", "original_index"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    return "—"


def _is_blank_value(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def sector_text(row) -> str:
    """Return the dashboard-facing Sector value.

    Dashboard terminology intentionally uses ``Sector`` rather than the model feature
    name ``sectorKey``.  The raw research datasource is not fully uniform across
    versions, therefore the display value is resolved in this order:
    explicit Sector -> industryKey -> sector -> industry -> sectorKey.

    ``industryKey`` is deliberately ahead of ``sectorKey`` because values such as
    ``medical-distribution`` are the operational sector labels requested for the UI,
    whereas ``sectorKey`` can be a broader model category such as ``healthcare``.
    """
    if row is None:
        return "—"
    idx = getattr(row, "index", [])
    for c in ["Sector", "industryKey", "sector", "industry", "sectorKey"]:
        if c in idx and not _is_blank_value(row[c]):
            return str(row[c]).strip()
    return "—"


def _ensure_sector_column(df):
    """Create/fill canonical ``Sector`` while preserving source columns."""
    if df is None:
        return None
    out = df.copy()
    if "Sector" not in out.columns:
        out["Sector"] = pd.NA
    for candidate in ["industryKey", "sector", "industry", "sectorKey"]:
        if candidate not in out.columns:
            continue
        missing = out["Sector"].apply(_is_blank_value)
        if not missing.any():
            break
        source = out[candidate]
        valid = ~source.apply(_is_blank_value)
        mask = missing & valid
        out.loc[mask, "Sector"] = source.loc[mask].astype(str).str.strip()
    return out


def segment_text(row) -> str:
    # Kept as a compatibility wrapper for result headers; UI wording is now Sector.
    return sector_text(row)


def risk_col_name(df):
    return detect_col(df, ["predicted_riskLevel", "y_pred", "predicted_risk", "riskLevel"])


def source_modified(path: Path | None):
    if not path or not path.exists():
        return "—"
    dt = datetime.fromtimestamp(path.stat().st_mtime)
    return dt.strftime("%d %b %Y, %H:%M")


def metric_value(metrics: dict, names, default=None):
    lower = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        if name in metrics:
            return metrics[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return default


def governance_code(path) -> str:
    p = str(path or "")
    if "priority_1" in p or p.startswith("P1"):
        return "P1"
    if "priority_2" in p or p.startswith("P2"):
        return "P2"
    if "priority_3" in p or p.startswith("P3"):
        return "P3"
    if "priority_4" in p or p.startswith("P4"):
        return "P4"
    if "priority_5" in p or p.startswith("P5"):
        return "P5"
    if "priority_6" in p or p.startswith("P6"):
        return "P6"
    return "—"


def governance_label(path) -> str:
    code = governance_code(path)
    return {
        "P1": "High Risk + Uncertain — immediate review",
        "P2": "High Risk + Certain — confirmatory review",
        "P3": "Medium Risk + Uncertain — boundary review",
        "P4": "Overconfident error — model evidence review",
        "P5": "General uncertainty — targeted manual review",
        "P6": "Regular monitoring",
    }.get(code, "Governance path unavailable")


def infer_governance_path(row) -> str:
    """Infer a production-safe governance path from prediction + uncertainty.

    P4 requires an observed actual label / correctness flag and therefore cannot be
    inferred for live production records that have no ground truth yet.
    """
    risk = risk_class(row.get("predicted_riskLevel", row.get("y_pred", "")))
    uncertain = bool(row.get("is_uncertain", False))
    if risk == "High Risk" and uncertain:
        return "priority_1_high_risk_uncertain_review"
    if risk == "High Risk":
        return "priority_2_high_risk_confirmatory_review"
    if risk == "Medium Risk" and uncertain:
        return "priority_3_medium_uncertain_boundary_review"
    if uncertain:
        return "priority_5_general_uncertainty_review"
    return "priority_6_regular_monitoring"


def resolved_governance_path(row) -> str:
    """Use pipeline governance path when present; otherwise infer it.

    pandas Series.get(default=...) does not use the default when a column exists
    but contains NaN. The previous dashboard therefore rendered an em dash for
    production rows that were not part of the Stage-4/final-test governance file.
    """
    try:
        value = row.get("governance_review_path", None)
    except Exception:
        value = None
    if value is not None and pd.notna(value):
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none", "null", "nat", "—", "-"}:
            return text
    return infer_governance_path(row)


def render_top_header(title, subtitle, meta_label, meta_value):
    left, right = st.columns([4, 1.25])
    with left:
        st.markdown(f'<div class="page-title">{esc(title)}</div><div class="page-sub">{esc(subtitle)}</div>', unsafe_allow_html=True)
    with right:
        st.markdown(f'<div class="top-meta">{esc(meta_label)}<br><span class="mono">{esc(meta_value)}</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="top-rule"></div>', unsafe_allow_html=True)


def confidence_bar_html(prob, confidence_text):
    p = max(0.0, min(1.0, to_float(prob, 0.0)))
    return (
        f'<div class="confidence-wrap">'
        f'<div class="confidence-row"><span class="confidence-label">Confidence Level</span>'
        f'<span class="confidence-text">{esc(confidence_text)} · {p:.2f}</span></div>'
        f'<div class="confidence-track"><div class="confidence-marker" style="left:calc({p*100:.1f}% - 1px)"></div></div>'
        f'<div class="confidence-ticks"><span>Low</span><span>Medium</span><span>High</span></div>'
        f'</div>'
    )


def render_confidence_bar(prob, confidence_text):
    st.markdown(confidence_bar_html(prob, confidence_text), unsafe_allow_html=True)


def probability_value(row, risk):
    candidates = {
        "Low Risk": ["proba_Low Risk", "P(Low Risk)", "p_Low Risk", "proba_Low", "p_low", "prob_low", "probability_low", "P(Low)"],
        "Medium Risk": ["proba_Medium Risk", "P(Medium Risk)", "p_Medium Risk", "proba_Medium", "p_medium", "prob_medium", "probability_medium", "P(Medium)"],
        "High Risk": ["proba_High Risk", "P(High Risk)", "p_High Risk", "proba_High", "p_high", "prob_high", "probability_high", "P(High)"]
    }
    for col in candidates.get(risk, []):
        if col in row.index and pd.notna(row[col]):
            return max(0.0, min(1.0, to_float(row[col], 0.0)))
    return 0.0


def render_predicted_risk_bars(row, predicted_risk):
    # Mirror the presentation order used in the requested reference chart.
    chart_order = ["High Risk", "Medium Risk", "Low Risk"]
    bars = []
    for risk in chart_order:
        val = probability_value(row, risk)
        height = max(8.0, val * 190.0)
        short = RISK_SHORT.get(risk, risk)
        highlight = ' hl' if risk == predicted_risk else ''
        badge = '<div class="pred-badge">predicted</div>' if risk == predicted_risk else ''
        bars.append(
            f'''<div class="pred-col">
            <div class="pred-value">{val:.2f}</div>
            <div class="pred-bar-wrap"><div class="pred-bar{highlight}" style="height:{height:.1f}px;background:{RISK_COLORS[risk]};"></div></div>
            <div class="pred-bar-label"><b>{esc(short)}</b><br>{fmt_pct(val)}{badge}</div>
            </div>'''
        )
    st.markdown(
        '<div class="pred-chart">'
        '<div class="pred-chart-title-row"><span class="pred-chart-title">Predicted Risk</span><span class="pred-chart-note">probability per risk class</span></div>'
        f'<div class="pred-chart-grid">{"".join(bars)}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_distribution(prod):
    rc = risk_col_name(prod)
    if rc is None:
        st.info("Kolom hasil klasifikasi tidak ditemukan pada production_prediction_output.")
        return
    risks = prod[rc].map(risk_class)
    total = len(prod)
    rows = []
    for r in RISK_ORDER:
        n = int((risks == r).sum())
        pct = n / total if total else 0
        rows.append(
            f'<div class="dist-row"><div>{esc(RISK_SHORT[r])}</div><div class="dist-track"><div class="dist-bar" style="width:{pct*100:.1f}%;background:{RISK_COLORS[r]}"></div></div><div class="dist-pct">{pct*100:.1f}%</div></div>'
        )
    st.markdown('<div class="panel"><div class="panel-title-row"><span class="panel-title">Distribusi Tier</span></div><div class="dist-wrap">'+''.join(rows)+'</div></div>', unsafe_allow_html=True)



def render_sector_key_risk_distribution(prod):
    """Show portfolio composition by the model's categorical ``sectorKey`` feature.

    Bars use a common count scale so both sector size and the mix of High/Medium/Low
    predictions are visible at once.  The underlying field remains ``sectorKey`` even
    though the business-facing customer tables use the friendlier ``Sector`` label.
    """
    if prod is None or prod.empty:
        return
    rc = risk_col_name(prod)
    if rc is None:
        return

    # sectorKey is the exact model attribute requested for this diagnostic chart.
    # It is populated by prepare_portfolio_df() from the datasource when Stage-5
    # prediction output contains only trace identifiers.
    sk_col = detect_col(prod, ["sectorKey", "sector_key", "SectorKey"])
    if sk_col is None:
        st.markdown(
            '<div class="panel" style="margin-top:1.1rem">'
            '<div class="panel-title-row"><span class="panel-title">Distribusi sectorKey per Risk Level</span>'
            '<span class="panel-note">atribut kategorikal model</span></div>'
            '<div class="notice" style="margin:1rem">sectorKey belum tersedia pada datasource/enrichment.</div></div>',
            unsafe_allow_html=True,
        )
        return

    d = pd.DataFrame({
        "sectorKey": prod[sk_col].astype("string").fillna("").str.strip(),
        "risk": prod[rc].map(risk_class),
    })
    d.loc[d["sectorKey"].eq(""), "sectorKey"] = "(missing)"
    d = d[d["risk"].isin(RISK_ORDER)]
    if d.empty:
        return

    pivot = (
        d.groupby(["sectorKey", "risk"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
    )
    pivot["__total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values(["__total", "High Risk"], ascending=[False, False])
    max_total = max(int(pivot["__total"].max()), 1)

    rows = []
    for sector, r in pivot.iterrows():
        total = int(r["__total"])
        h = int(r["High Risk"])
        m = int(r["Medium Risk"])
        l = int(r["Low Risk"])
        # Common x-scale across sectors: total bar length encodes sector size;
        # stacked segments encode risk composition inside that total.
        wh = h / max_total * 100.0
        wm = m / max_total * 100.0
        wl = l / max_total * 100.0
        rows.append(
            f'<div class="sector-risk-row">'
            f'<div class="sector-risk-label" title="{esc(sector)}">{esc(sector)}</div>'
            f'<div class="sector-risk-track">'
            f'<div class="sector-risk-seg" style="width:{wh:.3f}%;background:{RISK_COLORS["High Risk"]}"></div>'
            f'<div class="sector-risk-seg" style="width:{wm:.3f}%;background:{RISK_COLORS["Medium Risk"]}"></div>'
            f'<div class="sector-risk-seg" style="width:{wl:.3f}%;background:{RISK_COLORS["Low Risk"]}"></div>'
            f'</div>'
            f'<div class="sector-risk-values">H {h:,} · M {m:,} · L {l:,} · N {total:,}</div>'
            f'</div>'
        )

    legend = (
        '<div class="sector-risk-legend">'
        f'<span class="sector-risk-legend-item"><span class="sector-risk-dot" style="background:{RISK_COLORS["High Risk"]}"></span>High</span>'
        f'<span class="sector-risk-legend-item"><span class="sector-risk-dot" style="background:{RISK_COLORS["Medium Risk"]}"></span>Medium</span>'
        f'<span class="sector-risk-legend-item"><span class="sector-risk-dot" style="background:{RISK_COLORS["Low Risk"]}"></span>Low</span>'
        '</div>'
    )
    st.markdown(
        '<div class="panel" style="margin-top:1.1rem">'
        '<div class="panel-title-row"><span class="panel-title">Distribusi sectorKey per Risk Level</span></div>'
        f'<div class="sector-risk-wrap">{legend}{"".join(rows)}</div></div>',
        unsafe_allow_html=True,
    )


def merge_governance(prod, gov):
    if prod is None or gov is None or "original_index" not in prod.columns or "original_index" not in gov.columns:
        return prod
    cols = [c for c in ["original_index", "governance_review_path", "uncertainty_score", "sum_rw_ucfi_predicted_class", "max_rw_ucfi_predicted_class"] if c in gov.columns]
    if len(cols) <= 1:
        return prod
    g = gov[cols].drop_duplicates("original_index")
    return prod.merge(g, on="original_index", how="left")


def render_review_table(prod, gov):
    df = merge_governance(prod.copy(), gov)
    rc = risk_col_name(df)
    if rc is None:
        return
    high_col = probability_col(df, "High Risk")
    unc_col = detect_col(df, ["is_uncertain"])
    tmp = df.copy()
    tmp["_risk"] = tmp[rc].map(risk_class)
    if unc_col:
        tmp["_unc"] = normalize_bool(tmp[unc_col]).astype(int)
    else:
        tmp["_unc"] = 0
    tmp["_high"] = pd.to_numeric(tmp[high_col], errors="coerce") if high_col else np.nan
    tmp["_risk_priority"] = tmp["_risk"].map({"High Risk":0, "Medium Risk":1, "Low Risk":2}).fillna(3)
    tmp = tmp.sort_values(["_risk_priority", "_unc", "_high"], ascending=[True, False, False]).head(6)
    rows=[]
    for _, row in tmp.iterrows():
        path = resolved_governance_path(row)
        rows.append(
            "<tr>"
            f"<td>{esc(company_name(row))}</td>"
            f"<td>{esc(sector_text(row))}</td>"
            f"<td>{risk_pill(row[rc])}</td>"
            f"<td class='mono'>{fmt_prob(row.get(high_col)) if high_col else '—'}</td>"
            f"<td class='mono'>{esc(governance_code(path))}</td>"
            f"<td>{'Uncertain' if bool(row.get('_unc',0)) else 'Certain'}</td>"
            "</tr>"
        )
    table = f'''<div class="panel" style="margin-top:1.1rem"><div class="panel-title-row"><span class="panel-title">Perlu Tinjauan Segera</span><span class="panel-note">High Risk / uncertain diprioritaskan</span></div>
    <div class="table-wrap"><table class="mock-table"><thead><tr><th>Pelanggan</th><th>Sector</th><th>Tier</th><th>Prob. High</th><th>Review Path</th><th>Certainty</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div></div>'''
    st.markdown(table, unsafe_allow_html=True)


def render_portfolio(prod, gov, metrics, prod_path, data_root=None):
    render_top_header("Portfolio Risk Overview", "Ringkasan distribusi risk tier untuk seluruh pelanggan korporat aktif", "Diperbarui", source_modified(prod_path))
    if prod is None or prod.empty:
        st.warning("production_prediction_output belum tersedia. Copy output Stage 5 pipeline ke folder pipeline_output.")
        return
    display_prod = prepare_portfolio_df(prod, data_root) if data_root is not None else _ensure_sector_column(prod)
    rc = risk_col_name(display_prod)
    risks = display_prod[rc].map(risk_class) if rc else pd.Series([], dtype=str)
    total = len(display_prod)
    counts = {r:int((risks==r).sum()) for r in RISK_ORDER}
    def kpi(label,value,sub,klass=""):
        return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value {klass}">{value}</div><div class="kpi-sub">{sub}</div></div>'
    html_kpi = '<div class="kpi-grid">' + ''.join([
        kpi("Total pelanggan dinilai", f"{total:,}", "output production model"),
        kpi("Risk tinggi", f"{counts['High Risk']:,}", f"{counts['High Risk']/total*100:.1f}% dari portfolio" if total else "—", "high"),
        kpi("Risk sedang", f"{counts['Medium Risk']:,}", f"{counts['Medium Risk']/total*100:.1f}% dari portfolio" if total else "—", "medium"),
        kpi("Risk rendah", f"{counts['Low Risk']:,}", f"{counts['Low Risk']/total*100:.1f}% dari portfolio" if total else "—", "low"),
    ]) + '</div>'
    st.markdown(html_kpi, unsafe_allow_html=True)
    render_distribution(display_prod)
    render_sector_key_risk_distribution(display_prod)


def render_observation_portfolio(final_df, prod, metrics, paths, data_root=None):
    """Portfolio-style summary for the Stage-4 final-test observation set.

    This page is deliberately separated from the Stage-5 production portfolio so
    the global SHAP/RW-UCFI baseline is interpreted against the same final-test
    observations used by the research explainability pipeline.
    """
    source_path = paths.get("final_predictions") if paths else None
    render_top_header(
        "Portfolio Overview (Observasi)",
        "Ringkasan distribusi risk tier pada observasi final test dan global explainability penelitian",
        "Sumber",
        "Stage 4 · Final Test",
    )
    if final_df is None or final_df.empty:
        st.warning("final_test_predictions_with_uncertainty belum tersedia. Jalankan Stage 4 pipeline dan pastikan path output benar.")
        return

    obs = prepare_existing_customer_df(final_df, prod, data_root) if data_root is not None else final_df.copy().reset_index(drop=True)
    rc = risk_col_name(obs)
    risks = obs[rc].map(risk_class) if rc else pd.Series([], dtype=str)
    total = len(obs)
    counts = {r: int((risks == r).sum()) for r in RISK_ORDER}

    def kpi(label, value, sub, klass=""):
        return f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value {klass}">{value}</div><div class="kpi-sub">{sub}</div></div>'

    html_kpi = '<div class="kpi-grid">' + ''.join([
        kpi("Total observasi dinilai", f"{total:,}", "final test / Stage 4"),
        kpi("Risk tinggi", f"{counts['High Risk']:,}", f"{counts['High Risk']/total*100:.1f}% dari observasi" if total else "—", "high"),
        kpi("Risk sedang", f"{counts['Medium Risk']:,}", f"{counts['Medium Risk']/total*100:.1f}% dari observasi" if total else "—", "medium"),
        kpi("Risk rendah", f"{counts['Low Risk']:,}", f"{counts['Low Risk']/total*100:.1f}% dari observasi" if total else "—", "low"),
    ]) + '</div>'
    st.markdown(html_kpi, unsafe_allow_html=True)

    render_distribution(obs)
    render_sector_key_risk_distribution(obs)

    shap_global = load_optional(paths.get("shap_global")) if paths else None
    rw_global = load_optional(paths.get("rw_global")) if paths else None
    st.markdown(
        f'<div class="panel" style="margin-top:1.1rem">'
        f'<div class="panel-title-row"><span class="panel-title">Global Explainability — Final Test</span>'
        f'<span class="panel-note">SHAP Global &amp; RW-UCFI Global · N={total:,}</span></div></div>',
        unsafe_allow_html=True,
    )
    if shap_global is None and rw_global is None:
        st.info("Output Global SHAP dan Global RW-UCFI Stage 4 belum tersedia.")
        return

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_xai_bar_chart(
            shap_global,
            "mean_abs_shap_global",
            "Top 10 Global SHAP — Observasi Final Test",
            top_n=10,
            signed=False,
            rank_col="shap_global_rank",
        )
    with c2:
        render_xai_bar_chart(
            rw_global,
            "rw_ucfi_score_global",
            "Top 10 Global RW-UCFI — Observasi Final Test",
            top_n=10,
            signed=False,
            rank_col="rw_ucfi_global_rank",
        )



def search_customer_df(df, query):
    if df is None or df.empty:
        return df
    if not query.strip():
        return df.head(12)
    q=query.strip().lower()
    cols=[c for c in ["Company_Name","Company Name","Matched_Nama_CC","Matched Nama CC","longName","shortName","IDXCode","symbol","NIPNAS"] if c in df.columns]
    if not cols:
        return df.head(12)
    mask=pd.Series(False,index=df.index)
    for c in cols:
        mask |= df[c].astype(str).str.lower().str.contains(q, na=False, regex=False)
    return df.loc[mask].head(20)


def shap_for_index(shap_df, original_index):
    if shap_df is None or "original_index" not in shap_df.columns:
        return None
    idx_num = pd.to_numeric(shap_df["original_index"], errors="coerce")
    try:
        target=float(original_index)
        return shap_df[idx_num==target].copy()
    except Exception:
        return shap_df[shap_df["original_index"].astype(str)==str(original_index)].copy()


def rw_for_index(rw_df, original_index):
    if rw_df is None or "original_index" not in rw_df.columns:
        return None
    idx_num=pd.to_numeric(rw_df["original_index"],errors="coerce")
    try:
        return rw_df[idx_num==float(original_index)].copy()
    except Exception:
        return rw_df[rw_df["original_index"].astype(str)==str(original_index)].copy()


def render_gauge(probability: float):
    """Render the compact Low–Medium–High probability gauge used on Existing Customer.

    The marker is clamped to [0, 1] so malformed probabilities cannot break the layout.
    """
    p = max(0.0, min(1.0, to_float(probability, 0.0)))
    left = p * 100.0
    st.markdown(
        f'''
        <div style="padding:.82rem 1.3rem 1.0rem 1.3rem;border-top:1px solid {BORDER};background:{PANEL};">
          <div style="display:flex;justify-content:space-between;font-size:0.682rem;color:#8b8f89;margin-bottom:.38rem;">
            <span>Low</span><span>Medium</span><span>High</span>
          </div>
          <div style="position:relative;height:12px;background:linear-gradient(to right,
                       {RISK_COLORS['Low Risk']} 0%, {RISK_COLORS['Low Risk']} 33.33%,
                       {RISK_COLORS['Medium Risk']} 33.33%, {RISK_COLORS['Medium Risk']} 66.66%,
                       {RISK_COLORS['High Risk']} 66.66%, {RISK_COLORS['High Risk']} 100%);">
            <div style="position:absolute;left:{left:.2f}%;top:-7px;transform:translateX(-50%);
                        font-family:'DM Mono',monospace;font-size:0.682rem;color:#2f3430;white-space:nowrap;">{p:.2f}</div>
            <div style="position:absolute;left:{left:.2f}%;top:-3px;width:2px;height:18px;
                        background:#202522;transform:translateX(-1px);"></div>
          </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )


def render_driver_rows(df, value_col, label_col="original_column", topn=5):
    if df is None or df.empty or value_col not in df.columns:
        st.markdown('<div class="notice">Penjelasan lokal tidak tersedia untuk customer ini.</div>', unsafe_allow_html=True)
        return
    d=df.copy()
    d[value_col]=pd.to_numeric(d[value_col],errors="coerce")
    if "local_rank" in d.columns:
        d=d.sort_values("local_rank")
    elif "rw_ucfi_rank" in d.columns:
        d=d.sort_values("rw_ucfi_rank")
    else:
        d=d.reindex(d[value_col].abs().sort_values(ascending=False).index)
    d=d.head(topn)
    mx=max(float(d[value_col].abs().max()),1e-12)
    rows=[]
    for _,r in d.iterrows():
        v=to_float(r[value_col],0.0)
        width=min(100,abs(v)/mx*100)
        label=r.get(label_col,r.get("feature","feature"))
        bar=f'<div class="driver-pos" style="width:{width:.1f}%"></div>' if v>=0 else f'<div class="driver-neg" style="width:{width:.1f}%"></div>'
        rows.append(f'<div class="driver-row"><div class="driver-label">{esc(label)}</div><div class="driver-track">{bar}</div><div class="driver-value">{v:+.3f}</div></div>')
    st.markdown('<div class="explain-body">'+''.join(rows)+'</div>', unsafe_allow_html=True)


def _normalise_context_key(series, key):
    """Normalise join keys without changing the displayed source values."""
    if key == "original_index":
        num = pd.to_numeric(series, errors="coerce")
        return num.map(lambda x: "" if pd.isna(x) else str(int(x)) if float(x).is_integer() else str(float(x)))
    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.upper()
    )


def _fill_context_columns(base_df, lookup_df, columns):
    """Fill context using every available identifier, not only original_index.

    Stage-5 output contains trace identifiers but does not export sector/industry
    fields.  ``original_index`` is useful when the exact modelling dataframe is
    loaded, but it can differ when a raw/source workbook has been regenerated.
    Therefore missing values are filled sequentially by original_index, IDXCode,
    symbol, and NIPNAS.  This specifically prevents valid sectors from becoming
    blank merely because the row index does not match.
    """
    if base_df is None or base_df.empty or lookup_df is None or lookup_df.empty:
        return base_df
    out = base_df.copy()
    lookup = lookup_df.copy()
    keys = ["original_index", "IDXCode", "symbol", "NIPNAS"]

    for key in keys:
        if key not in out.columns or key not in lookup.columns:
            continue
        # Never select the join key twice. Some context lists also contain
        # IDXCode/symbol/NIPNAS; selecting [key] + available with the same
        # name twice makes ``lk[key]`` return a DataFrame instead of a Series,
        # which then breaks the ``.str`` accessor in _normalise_context_key.
        available = []
        for c in columns:
            if c == key or c not in lookup.columns or c in available:
                continue
            available.append(c)
        if not available:
            continue

        base_key = _normalise_context_key(out[key], key)
        selected_cols = [key] + available
        lk = lookup.loc[:, selected_cols].copy()
        # Defensive guard for workbooks/dataframes that themselves contain
        # duplicated column labels. Keep the first copy so lk[key] is Series.
        lk = lk.loc[:, ~lk.columns.duplicated(keep="first")]
        lk["__join_key"] = _normalise_context_key(lk[key], key)
        lk = lk[lk["__join_key"].astype(str).str.len() > 0]
        lk = lk.drop_duplicates(subset=["__join_key"], keep="first")
        if lk.empty:
            continue

        for col in available:
            mapper = lk.set_index("__join_key")[col]
            mapped = base_key.map(mapper)
            if col not in out.columns:
                out[col] = mapped
            else:
                missing = out[col].apply(_is_blank_value)
                valid_mapped = ~mapped.apply(_is_blank_value)
                mask = missing & valid_mapped
                out.loc[mask, col] = mapped.loc[mask]

    return out


def _load_raw_context(data_root):
    try:
        ref = discover_reference_dataset(data_root)
        if ref is None:
            return None
        raw = load_table(str(ref)).copy()
        if "original_index" not in raw.columns:
            raw = raw.reset_index().rename(columns={"index": "original_index"})
        return raw
    except Exception:
        return None


def prepare_portfolio_df(prod, data_root):
    """Enrich Stage-5 portfolio output with the operational ``Sector`` field.

    The current pipeline trace export intentionally contains identifiers only; the
    sector metadata is recovered from the datasource by identifier.
    """
    if prod is None:
        return None
    out = prod.copy().reset_index(drop=True)
    context_cols = [
        "Sector", "sector", "sectorKey", "industryKey", "industry", "city",
        "Company_Name", "Company Name", "Matched_Nama_CC", "Matched Nama CC",
        "longName", "shortName", "IDXCode", "symbol", "NIPNAS",
    ]
    raw = _load_raw_context(data_root)
    if raw is not None:
        out = _fill_context_columns(out, raw, context_cols)
    return _ensure_sector_column(out)


def prepare_existing_customer_df(source_df, prod, data_root):
    """Enrich Stage-4 rows with company context and canonical ``Sector``."""
    if source_df is None:
        return None
    out = source_df.copy().reset_index(drop=True)
    out["_existing_uid"] = np.arange(len(out), dtype=int)
    context_cols = [
        "Sector", "sector", "sectorKey", "industryKey", "industry", "city",
        "Company_Name", "Company Name", "Matched_Nama_CC", "Matched Nama CC",
        "longName", "shortName", "IDXCode", "symbol", "NIPNAS",
    ]
    out = _fill_context_columns(out, prod, context_cols)

    raw = _load_raw_context(data_root)
    if raw is not None:
        out = _fill_context_columns(out, raw, context_cols)

    return _ensure_sector_column(out)


def filter_existing_customers(df, query, risk_filter="All", sector_filter="All"):
    if df is None or df.empty:
        return df
    out = df.copy()
    rc = risk_col_name(out)
    if risk_filter != "All" and rc:
        out = out[out[rc].map(risk_class) == risk_filter]
    if sector_filter != "All" and "Sector" in out.columns:
        out = out[out["Sector"].astype(str).str.strip() == str(sector_filter)]
    q = str(query or "").strip().lower()
    if q:
        cols = [c for c in [
            "Company_Name", "Company Name", "Matched_Nama_CC", "Matched Nama CC",
            "longName", "shortName", "IDXCode", "symbol", "NIPNAS", "Sector", "industryKey", "sectorKey"
        ] if c in out.columns]
        if cols:
            mask = pd.Series(False, index=out.index)
            for c in cols:
                mask |= out[c].astype(str).str.lower().str.contains(q, na=False, regex=False)
            out = out.loc[mask]
    return out


def render_existing_detail(selected, source_df, shap_df, rw_df, gov_df):
    """Use the same result hierarchy as New Customer Prediction."""
    rc = risk_col_name(source_df)
    risk = risk_class(selected.get(rc, "—"))
    p = probability_value(selected, risk)
    if p <= 0:
        pcol = probability_col(source_df, risk)
        p = to_float(selected.get(pcol), to_float(selected.get("max_proba"), 0.0)) if pcol else to_float(selected.get("max_proba"), 0.0)
    p = max(0.0, min(1.0, to_float(p, 0.0)))
    color = RISK_COLORS.get(risk, "#555")
    uncertain = bool(selected.get("is_uncertain", False))
    status = "uncertain" if uncertain else "certain"
    confidence = "rendah" if uncertain else ("tinggi" if p >= .80 else "sedang")

    # Header + blue confidence bar: same treatment as New Customer Prediction.
    st.markdown(
        f'<div class="panel prediction-summary"><div class="detail-head"><div>'
        f'<div class="detail-name">{esc(company_name(selected))}</div>'
        f'<div class="detail-meta">{esc(customer_code(selected))} &nbsp;&nbsp; {esc(segment_text(selected))}<br>certainty: {status}</div></div>'
        f'<div class="detail-risk"><div class="detail-risk-label">Risk Tier</div>'
        f'<div class="detail-risk-value" style="color:{color}">{esc(RISK_SHORT.get(risk,risk))}</div>'
        f'<div class="detail-risk-prob">P = {p:.2f}</div></div></div>'
        f'<div class="prediction-confidence-section">{confidence_bar_html(p, confidence)}</div></div>',
        unsafe_allow_html=True,
    )

    # Same three-class probability chart as Menu 03.
    render_predicted_risk_bars(selected, risk)

    orig = selected.get("original_index", None)
    st.markdown('<div class="panel-title-row"><span class="panel-title">Faktor Pendorong (SHAP)</span><span class="panel-note">local · predicted class</span></div>', unsafe_allow_html=True)
    sd = shap_for_index(shap_df, orig) if orig is not None else None
    render_driver_rows(sd, "shap_value", topn=10)

    st.markdown('<div class="panel-title-row"><span class="panel-title">RW-UCFI Local</span><span class="panel-note">risk-weighted + uncertainty-conditioned</span></div>', unsafe_allow_html=True)
    rd = rw_for_index(rw_df, orig) if orig is not None else None
    value_col = detect_col(rd, [
        "rw_ucfi_score_entity", "rw_ucfi_score", "rw_ucfi_score_predicted_class", "rw_ucfi_value"
    ]) if rd is not None else None
    if value_col:
        # Pipeline entity-level output uses entity_rank, not rw_ucfi_rank.
        if rd is not None and "entity_rank" in rd.columns and "rw_ucfi_rank" not in rd.columns:
            rd = rd.copy()
            rd["rw_ucfi_rank"] = rd["entity_rank"]
        render_driver_rows(rd, value_col, topn=10)
    else:
        st.markdown('<div class="notice">RW-UCFI entity-level tidak tersedia untuk customer ini.</div>', unsafe_allow_html=True)

    gd = None
    if gov_df is not None and orig is not None and "original_index" in gov_df.columns:
        gd = rw_for_index(gov_df, orig)
    if gd is not None and not gd.empty:
        path = resolved_governance_path(gd.iloc[0])
    else:
        path = resolved_governance_path(selected)
    st.markdown(
        '<div class="panel-title-row"><span class="panel-title">Governance Review</span><span class="panel-note">P1–P6 review path</span></div>'
        f'<div class="path-card"><div class="path-code">{esc(governance_code(path))}</div>'
        f'<div class="path-name">{esc(governance_label(path))}</div>'
        f'<div class="path-desc">Review path mengikuti kombinasi risk tier dan uncertainty pada pipeline.</div></div>',
        unsafe_allow_html=True,
    )


def _canonical_observation_index(value):
    """Canonicalize final-test indices so CSV/Excel floats match NPZ integer indices."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    try:
        num = float(value)
        if np.isfinite(num):
            return str(int(num)) if num.is_integer() else format(num, ".15g")
    except Exception:
        pass
    return str(value).strip()


def filtered_global_xai(filtered_df, tensor_path):
    """Aggregate SHAP and RW-UCFI over all observations passing the active filters.

    This uses the exact Stage-4 formulas:
      SHAP global   = mean(abs(SHAP), samples, classes)
      RW-UCFI global = mean(RW-UCFI tensor, samples, classes)

    Pagination is intentionally ignored; ``filtered_df`` must be the full filtered
    population, not only the visible 10-row page.
    """
    if filtered_df is None or filtered_df.empty or tensor_path is None:
        return None, None, 0, ""
    path = Path(tensor_path)
    if not path.exists():
        return None, None, 0, f"Artifact tidak ditemukan: {path.name}"
    if "original_index" not in filtered_df.columns:
        return None, None, 0, "Kolom original_index tidak tersedia pada data observasi."

    artifact = load_rw_ucfi_tensor_artifact(str(path))
    if not artifact:
        return None, None, 0, f"Artifact {path.name} tidak dapat dibaca."

    required = {"abs_shap_values", "rw_ucfi_tensor", "feature_names", "explained_indices"}
    missing = required.difference(artifact)
    if missing:
        return None, None, 0, f"Artifact {path.name} tidak lengkap: {', '.join(sorted(missing))}."

    abs_shap = np.asarray(artifact["abs_shap_values"], dtype=float)
    rw_tensor = np.asarray(artifact["rw_ucfi_tensor"], dtype=float)
    features = [str(x) for x in np.asarray(artifact["feature_names"]).tolist()]
    originals = artifact.get("original_feature_names")
    original_names = [str(x) for x in np.asarray(originals).tolist()] if originals is not None else features
    explained = np.asarray(artifact["explained_indices"]).tolist()

    if abs_shap.ndim != 3 or rw_tensor.ndim != 3:
        return None, None, 0, "Dimensi tensor SHAP/RW-UCFI tidak sesuai format Stage 4."
    if abs_shap.shape != rw_tensor.shape:
        return None, None, 0, "Dimensi SHAP dan RW-UCFI tensor tidak konsisten."
    if abs_shap.shape[0] != len(explained) or abs_shap.shape[1] != len(features):
        return None, None, 0, "Metadata tensor tidak konsisten dengan dimensi array."

    index_to_pos = {}
    for pos, idx in enumerate(explained):
        key = _canonical_observation_index(idx)
        if key and key not in index_to_pos:
            index_to_pos[key] = pos

    wanted = []
    seen = set()
    for idx in filtered_df["original_index"].tolist():
        key = _canonical_observation_index(idx)
        if key in index_to_pos and key not in seen:
            wanted.append(index_to_pos[key])
            seen.add(key)

    if not wanted:
        return None, None, 0, "Tidak ada original_index hasil filter yang cocok dengan explained_indices Stage 4."

    shap_subset = abs_shap[wanted, :, :]
    rw_subset = rw_tensor[wanted, :, :]

    # Same aggregation as pipeline Cell 09.10 and Cell 10.7.
    shap_score = shap_subset.mean(axis=0).mean(axis=1)
    rw_score = rw_subset.mean(axis=(0, 2))

    shap_global = pd.DataFrame({
        "feature": features,
        "original_column": original_names,
        "mean_abs_shap_global": shap_score,
    }).sort_values("mean_abs_shap_global", ascending=False).reset_index(drop=True)
    shap_global["shap_global_rank"] = np.arange(1, len(shap_global) + 1)

    rw_global = pd.DataFrame({
        "feature": features,
        "original_column": original_names,
        "rw_ucfi_score_global": rw_score,
    }).sort_values("rw_ucfi_score_global", ascending=False).reset_index(drop=True)
    rw_global["rw_ucfi_global_rank"] = np.arange(1, len(rw_global) + 1)

    unmatched = len(filtered_df) - len(wanted)
    note = "" if unmatched == 0 else f"{unmatched:,} observasi hasil filter tidak memiliki pasangan tensor Stage 4."
    return shap_global, rw_global, len(wanted), note


def render_filtered_global_xai(filtered_df, tensor_path):
    shap_global, rw_global, n_matched, note = filtered_global_xai(filtered_df, tensor_path)
    total_filtered = 0 if filtered_df is None else len(filtered_df)

    st.markdown(
        f'<div class="panel" style="margin-top:1.35rem">'
        f'<div class="panel-title-row"><span class="panel-title">Explainability — Filtered Population</span>'
        f'<span class="panel-note">N={n_matched:,} dari {total_filtered:,} customer terfilter</span></div></div>',
        unsafe_allow_html=True,
    )

    if shap_global is None or rw_global is None:
        message = note or "Filtered Global SHAP/RW-UCFI belum dapat dihitung."
        st.info(message + " Pastikan output Stage 4 `rw_ucfi_analysis/rw_ucfi_tensor.npz` tersedia.")
        return

    if note:
        st.caption(note)

    # N=1 is mathematically the same aggregation formula, but it is no longer a
    # population-level summary, so make the label explicit.
    if n_matched == 1:
        shap_title = "Filtered SHAP Aggregate — N=1"
        rw_title = "Filtered RW-UCFI Aggregate — N=1"
    else:
        shap_title = f"Top 10 Filtered Global SHAP — N={n_matched:,}"
        rw_title = f"Top 10 Filtered Global RW-UCFI — N={n_matched:,}"

    c1, c2 = st.columns(2, gap="large")
    with c1:
        render_xai_bar_chart(
            shap_global,
            "mean_abs_shap_global",
            shap_title,
            top_n=10,
            signed=False,
            rank_col="shap_global_rank",
        )
    with c2:
        render_xai_bar_chart(
            rw_global,
            "rw_ucfi_score_global",
            rw_title,
            top_n=10,
            signed=False,
            rank_col="rw_ucfi_global_rank",
        )


def render_existing_customer(prod, final_df, shap_df, rw_df, gov_df, data_root=None, tensor_path=None):
    # Stage-4 final-test set is preferred because every displayed local explanation is model-consistent.
    raw_source = final_df if final_df is not None and not final_df.empty else prod
    source_name = "final-test explainability set" if raw_source is final_df else "production portfolio"
    render_top_header("Existing Customer – Risk Profile", "Cari dan filter pelanggan untuk melihat klasifikasi serta bukti model", "Sumber", source_name)
    if raw_source is None or raw_source.empty:
        st.warning("Data customer belum tersedia. Jalankan Stage 4/Stage 5 pipeline dan pastikan path output benar.")
        return

    source_df = prepare_existing_customer_df(raw_source, prod, data_root) if data_root is not None else raw_source.copy().reset_index(drop=True)
    if "_existing_uid" not in source_df.columns:
        source_df["_existing_uid"] = np.arange(len(source_df), dtype=int)
    rc = risk_col_name(source_df)

    # Search + two requested filters.
    f1, f2, f3 = st.columns([2.2, 1.15, 1.35], gap="small")
    with f1:
        query = st.text_input("Cari pelanggan", placeholder="Nama perusahaan / IDXCode / NIPNAS", key="existing_search")
    risk_options = ["All", "High Risk", "Medium Risk", "Low Risk"]
    with f2:
        risk_filter = st.selectbox("Risk Level", risk_options, key="existing_risk_filter")
    sectors = []
    if "Sector" in source_df.columns:
        vals = source_df["Sector"].dropna().astype(str).str.strip()
        sectors = sorted([x for x in vals.unique().tolist() if x and x.lower() not in {"nan", "none"}], key=lambda x: x.lower())
    with f3:
        sector_filter = st.selectbox("Sector", ["All"] + sectors, key="existing_sector_filter")

    filtered = filter_existing_customers(source_df, query, risk_filter, sector_filter)
    total_all = len(source_df)
    total_filtered = 0 if filtered is None else len(filtered)
    if filtered is None or filtered.empty:
        st.info("Tidak ada customer yang sesuai dengan filter.")
        return

    # Reset pagination when the effective filter changes.
    signature = (str(query or ""), risk_filter, sector_filter)
    if st.session_state.get("existing_filter_signature") != signature:
        st.session_state["existing_filter_signature"] = signature
        st.session_state["existing_page"] = 1

    page_size = 10
    total_pages = max(1, int(np.ceil(total_filtered / page_size)))
    current_page = int(st.session_state.get("existing_page", 1) or 1)
    if current_page > total_pages:
        st.session_state["existing_page"] = total_pages
    if current_page < 1:
        st.session_state["existing_page"] = 1

    # Keep selection valid after filtering.
    filtered_uids = filtered["_existing_uid"].astype(int).tolist()
    selected_uid = st.session_state.get("existing_selected_uid")
    if selected_uid not in filtered_uids:
        st.session_state["existing_selected_uid"] = int(filtered_uids[0])

    left, right = st.columns([1.0, 1.19], gap="medium")
    with left:
        head_l, head_r = st.columns([2.4, 1.0])
        with head_l:
            shown_text = f"{total_filtered:,} filtered" if total_filtered != total_all else f"{total_all:,} all"
            st.markdown(f'<div class="existing-list-head"><span class="panel-title">Customer Data</span><span class="existing-list-count">{shown_text}</span></div>', unsafe_allow_html=True)
        with head_r:
            if total_pages > 1:
                st.selectbox("Halaman", list(range(1, total_pages + 1)), key="existing_page", format_func=lambda x: f"Page {x}/{total_pages}")
            else:
                st.markdown('<div class="pagination-note">Page 1/1</div>', unsafe_allow_html=True)

        page = int(st.session_state.get("existing_page", 1) or 1)
        start_idx = (page - 1) * page_size
        stop_idx = min(start_idx + page_size, total_filtered)
        page_df = filtered.iloc[start_idx:stop_idx]
        st.caption(f"Menampilkan {start_idx + 1:,}–{stop_idx:,} dari {total_filtered:,} customer")

        for _, row in page_df.iterrows():
            uid = int(row["_existing_uid"])
            is_selected = uid == int(st.session_state.get("existing_selected_uid", uid))
            with st.container(border=True):
                c1, c2, c3 = st.columns([2.8, 1.45, .72], gap="small")
                with c1:
                    st.markdown(f'<div class="existing-card-title">{esc(company_name(row))}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="existing-card-meta">{esc(customer_code(row))}</div>', unsafe_allow_html=True)
                    if is_selected:
                        st.markdown('<div class="existing-selected">selected</div>', unsafe_allow_html=True)
                with c2:
                    sector = sector_text(row)
                    st.markdown(f'<div class="existing-card-sector">{esc(sector)}</div>', unsafe_allow_html=True)
                    st.markdown(risk_pill(row.get(rc, "—")), unsafe_allow_html=True)
                with c3:
                    if st.button("Lihat", key=f"existing_open_{uid}", use_container_width=True):
                        st.session_state["existing_selected_uid"] = uid

    with right:
        uid = int(st.session_state.get("existing_selected_uid", filtered_uids[0]))
        selected_rows = source_df[source_df["_existing_uid"].astype(int) == uid]
        if selected_rows.empty:
            selected = filtered.iloc[0]
            st.session_state["existing_selected_uid"] = int(selected["_existing_uid"])
        else:
            selected = selected_rows.iloc[0]
        render_existing_detail(selected, source_df, shap_df, rw_df, gov_df)

    # Filter-aware global explainability uses ALL rows passing Search/Risk/Sector,
    # independent of the visible pagination page.
    render_filtered_global_xai(filtered, tensor_path)


def find_feature(bundle, candidates):
    selected=list(bundle.get("selected_features",[]))
    lower={str(x).lower():x for x in selected}
    for c in candidates:
        if c in selected: return c
        if c.lower() in lower: return lower[c.lower()]
    return None


def discover_reference_dataset(data_root: str | Path):
    """Find the pipeline raw dataset used as the Kernel SHAP reference distribution."""
    root=Path(data_root).expanduser()
    if root.is_file() and root.suffix.lower() in {".xlsx",".xls",".csv"}:
        return root
    if not root.exists():
        return None
    preferred=[
        "yfinance_bud_merged_labelled.xlsx",
        "yfinance_bud_merged_labelled_newlabel.xlsx",
    ]
    for name in preferred:
        p=root/name
        if p.exists(): return p
    candidates=[]
    for pat in ["yfinance_bud_merged_labelled*.xlsx","yfinance_bud_merged_labelled*.csv"]:
        candidates.extend(root.glob(pat))
    return sorted(candidates)[0] if candidates else None


def infer_data_source_from_output(output_root: str | Path):
    """Infer <pipeline_base>/data from <pipeline_base>/output when dashboard is stored elsewhere."""
    try:
        out = resolve_output_root(output_root)
        candidates = [
            out.parent / "data",               # .../Semhas_New_Pipeline/output -> .../Semhas_New_Pipeline/data
            out / "data",
        ]
        for candidate in candidates:
            ref = discover_reference_dataset(candidate)
            if ref is not None:
                return candidate
    except Exception:
        pass
    return None


def resolve_data_source(user_value: str | Path, output_root: str | Path):
    """Resolve an independent datasource path, then fall back to the pipeline sibling data folder."""
    raw = Path(user_value).expanduser() if str(user_value).strip() else None
    if raw is not None and discover_reference_dataset(raw) is not None:
        return raw
    inferred = infer_data_source_from_output(output_root)
    if inferred is not None:
        return inferred
    return raw if raw is not None else Path(user_value)


def _norm_feature_name(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def find_features_normalized(bundle, candidates):
    """Return selected-feature names matching candidate names after punctuation/underscore normalization."""
    selected=list(bundle.get("selected_features", []))
    wanted={_norm_feature_name(c) for c in candidates}
    return [f for f in selected if _norm_feature_name(f) in wanted]


def feature_targets(bundle, candidates, fallback):
    """Resolve one or more selected feature names; use fallback raw name if current bundle has no match."""
    matches=find_features_normalized(bundle, candidates)
    return matches if matches else [fallback]


@st.cache_data(show_spinner=False)
def _sector_values_from_file(path_str):
    """Read distinct sectorKey values from the pipeline database/reference file."""
    p=Path(path_str)
    if not p.exists():
        return []
    try:
        if p.suffix.lower()==".csv":
            df=pd.read_csv(p, usecols=lambda c: _norm_feature_name(c)=="sectorkey")
        else:
            # Read only the header first so Excel usecols can be resolved robustly.
            hdr=pd.read_excel(p, nrows=0)
            cols=[c for c in hdr.columns if _norm_feature_name(c)=="sectorkey"]
            if not cols:
                return []
            df=pd.read_excel(p, usecols=cols)
        col=next((c for c in df.columns if _norm_feature_name(c)=="sectorkey"), None)
        if col is None:
            return []
        vals=(df[col].dropna().astype(str).str.strip())
        vals=vals[vals.ne("")]
        return sorted(vals.unique().tolist(), key=lambda x: x.lower())
    except Exception:
        return []


def get_sector_options(data_root, production_predictions_path=None, output_root=None):
    """Load sectorKey choices from an independent datasource; pipeline and dashboard may be in different folders."""
    candidates=[]
    checked=[]

    def add_candidate(value):
        if value is None:
            return
        p=Path(value).expanduser()
        ref=discover_reference_dataset(p)
        if ref is not None and ref not in candidates:
            candidates.append(ref)
        checked.append(str(p))

    # 1) Explicit datasource path from sidebar / B2B_DATA_ROOT.
    add_candidate(data_root)

    # 2) Auto-infer sibling data folder from the selected pipeline output path.
    if output_root is not None:
        inferred=infer_data_source_from_output(output_root)
        if inferred is not None:
            add_candidate(inferred)

    # 3) Stage-5 prediction output fallback, if it happens to contain sectorKey.
    if production_predictions_path is not None:
        p=Path(production_predictions_path)
        checked.append(str(p))
        if p.exists() and p not in candidates:
            candidates.append(p)

    for p in candidates:
        values=_sector_values_from_file(str(p))
        if values:
            return values, p, checked
    return [], None, checked


def render_xai_bar_chart(df, value_col, title, top_n=10, signed=False, rank_col=None):
    if df is None or df.empty or value_col not in df.columns:
        st.info("Data chart tidak tersedia.")
        return
    d=df.copy()
    d[value_col]=pd.to_numeric(d[value_col],errors="coerce")
    d=d.dropna(subset=[value_col])
    if rank_col and rank_col in d.columns:
        d=d.sort_values(rank_col).head(int(top_n))
    elif signed:
        d=d.reindex(d[value_col].abs().sort_values(ascending=False).index).head(int(top_n))
    else:
        d=d.sort_values(value_col,ascending=False).head(int(top_n))
    d=d.reset_index(drop=True)
    d["rank"] = np.arange(1,len(d)+1)
    label_col="original_column" if "original_column" in d.columns else "feature"
    d["feature_label"]=d[label_col].astype(str).map(lambda x: x if len(x)<=48 else x[:45]+"...")
    d["score"]=d[value_col].astype(float)
    color = (
        {"condition":{"test":"datum.score >= 0","value":"#aa4935"},"value":"#477b5f"}
        if signed else {"value":"#3f6272"}
    )
    spec={
        "height": max(220, 29*len(d)),
        "mark":{"type":"bar","cornerRadiusEnd":2},
        "encoding":{
            "y":{
                "field":"feature_label","type":"nominal",
                "sort":{"field":"rank","order":"ascending"},
                "title":None,
                "axis":{"labelLimit":310,"labelFontSize":11}
            },
            "x":{"field":"score","type":"quantitative","title":title,"axis":{"format":".3g"}},
            "color": color,
            "tooltip":[
                {"field":"feature_label","type":"nominal","title":"Feature"},
                {"field":"score","type":"quantitative","title":"Score","format":".6f"}
            ]
        },
        "config":{"view":{"stroke":None},"axis":{"gridColor":"#ebe7de","domainColor":"#cfcac0"}}
    }
    st.markdown(f"**{title}**")
    st.vega_lite_chart(d[["feature_label","score","rank"]],spec,use_container_width=True)


def render_live_xai_results(xai_results, prediction_df, paths, top_n=10, selected_position=None):
    if not xai_results:
        return
    meta=xai_results.get("metadata",{})
    summary=xai_results.get("sample_summary",pd.DataFrame())
    st.markdown('<div class="top-rule"></div>',unsafe_allow_html=True)
    st.markdown('<div class="page-title" style="font-size:1.485rem">Live Explainability — SHAP & RW-UCFI</div>',unsafe_allow_html=True)
    st.caption(
        f"Kernel SHAP · {meta.get('n_explained_rows',0)} baris dijelaskan · "
        f"background {meta.get('background_rows',0)} · nsamples {meta.get('nsamples',0)}. "
        "RW-UCFI adalah post-hoc governance prioritization dan tidak mengubah prediksi/probabilitas."
    )

    tabs=st.tabs(["Local — per customer","Global — current run","Global baseline penelitian"])
    with tabs[0]:
        if summary is None or summary.empty:
            st.info("Tidak ada local explanation.")
        else:
            explained_positions=[int(x) for x in summary["sample_position"].tolist()]
            if selected_position is None:
                pos=explained_positions[0]
            else:
                pos=int(selected_position)
            if pos not in explained_positions:
                st.info(
                    f"Customer #{pos+1} belum termasuk {len(explained_positions)} baris yang dijelaskan pada run ini. "
                    "Naikkan 'Jumlah baris dijelaskan' dan jalankan ulang prediction untuk memperoleh Local SHAP/RW-UCFI."
                )
            else:
                st.caption(f"Local explanation mengikuti customer terpilih di panel hasil: #{pos+1} · {company_name(prediction_df.iloc[pos])}")
                local_shap=xai_results["local_shap"]
                local_rw=xai_results["local_rw_ucfi"]
                ls=local_shap[local_shap["sample_position"]==pos].copy()
                lr=local_rw[local_rw["sample_position"]==pos].copy()
                c1,c2=st.columns(2,gap="large")
                with c1:
                    render_xai_bar_chart(ls,"shap_value","Top Local SHAP — predicted class",top_n,True,"local_shap_rank")
                with c2:
                    render_xai_bar_chart(lr,"rw_ucfi_score_entity","Top Local RW-UCFI",top_n,False,"local_rw_ucfi_rank")
                sr=summary[summary["sample_position"]==pos].iloc[0]
                st.caption(
                    f"Uncertainty score {float(sr['uncertainty_score']):.3f} · multiplier {float(sr['uncertainty_multiplier']):.3f} · "
                    f"max probability {float(sr['max_proba']):.3f} · margin {float(sr['margin']):.3f} · entropy {float(sr['entropy_norm']):.3f}."
                )

    with tabs[1]:
        st.caption("Global live = rata-rata importance atas baris yang dijelaskan pada run ini, bukan seluruh populasi training.")
        c1,c2=st.columns(2,gap="large")
        with c1:
            render_xai_bar_chart(xai_results["global_shap"],"mean_abs_shap_global","Top Global SHAP — live run",top_n,False,"shap_global_rank")
        with c2:
            render_xai_bar_chart(xai_results["global_rw_ucfi"],"rw_ucfi_score_global","Top Global RW-UCFI — live run",top_n,False,"rw_ucfi_global_rank")

    with tabs[2]:
        baseline_shap=load_optional(paths.get("shap_global"))
        baseline_rw=load_optional(paths.get("rw_global"))
        if baseline_shap is None and baseline_rw is None:
            st.info("Global baseline Stage 4 belum tersedia. Copy output explainability_shap dan rw_ucfi_analysis dari pipeline.")
        else:
            st.caption("Baseline penelitian menggunakan global importance dari final-test explainability pipeline Stage 4.")
            c1,c2=st.columns(2,gap="large")
            with c1:
                render_xai_bar_chart(baseline_shap,"mean_abs_shap_global","Top Global SHAP — final test",top_n,False,"shap_global_rank")
            with c2:
                render_xai_bar_chart(baseline_rw,"rw_ucfi_score_global","Top Global RW-UCFI — final test",top_n,False,"rw_ucfi_global_rank")

    st.download_button(
        "Download hasil SHAP & RW-UCFI",
        data=xai_results_to_excel_bytes(xai_results),
        file_name="new_customer_live_shap_rw_ucfi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=False,
    )


def execute_new_prediction(raw, bundle, xai_cfg, paths, data_root, background_upload=None):
    out,report=predict_from_dataframe(raw,bundle)
    xai=None; xai_error=None
    if xai_cfg and xai_cfg.get("enabled"):
        try:
            background_X=None; background_raw=None
            if background_upload is not None:
                background_raw=read_uploaded_table(background_upload)
            elif paths.get("xai_background") and paths["xai_background"].exists():
                background_X=cached_xai_background(str(paths["xai_background"]))
            else:
                ref_path=discover_reference_dataset(data_root)
                if ref_path is not None:
                    background_raw=load_table(str(ref_path))

            xai=compute_live_xai(
                explain_raw=raw,
                prediction_output=out,
                bundle=bundle,
                background_X=background_X,
                background_raw=background_raw,
                explain_rows=xai_cfg["explain_rows"],
                background_size=xai_cfg["background_size"],
                top_n=xai_cfg["top_n"],
                nsamples=xai_cfg["nsamples"],
                risk_weights=xai_cfg["risk_weights"],
                uncertainty_alpha=xai_cfg["alpha"],
                uncertain_flag_bonus=xai_cfg["flag_bonus"]
            )
        except Exception as e:
            xai_error=str(e)
    return out,report,xai,xai_error


def render_prediction_result(row, report, live_xai=None, sample_position=0):
    rc=risk_col_name(pd.DataFrame([row]))
    risk=risk_class(row.get(rc,"—"))
    p=probability_value(row, risk)
    if p <= 0:
        p=to_float(row.get("max_proba"),0.0)
    color=RISK_COLORS.get(risk,"#555")
    uncertain=bool(row.get("is_uncertain",False))
    confidence="rendah" if uncertain else ("tinggi" if p>=.80 else "sedang")
    confidence_html = confidence_bar_html(p, confidence)
    st.markdown(
        f'<div class="panel prediction-summary"><div class="detail-head"><div><div class="detail-name">{esc(company_name(row))}</div>'
        f'<div class="detail-meta">Belum tercatat di CRM</div></div>'
        f'<div class="detail-risk"><div class="detail-risk-label">Hasil Prediksi</div><div class="detail-risk-value" style="color:{color}">'
        f'{esc(RISK_SHORT.get(risk,risk))}</div><div class="detail-risk-prob">P = {p:.2f}</div></div></div>'
        f'<div class="prediction-confidence-section">{confidence_html}</div></div>',
        unsafe_allow_html=True
    )
    render_predicted_risk_bars(row, risk)
    st.markdown('<div class="panel-title-row"><span class="panel-title">Faktor Pendorong (SHAP)</span><span class="panel-note">live · predicted class</span></div>',unsafe_allow_html=True)
    if live_xai:
        d=live_xai["local_shap"]
        d=d[d["sample_position"]==int(sample_position)].copy()
        if not d.empty:
            d["local_rank"]=d["local_shap_rank"]
            render_driver_rows(d,"shap_value",topn=10)
        else:
            st.markdown('<div class="notice">SHAP belum dihitung untuk customer ini. Naikkan <b>Jumlah baris dijelaskan</b> lalu jalankan ulang prediction.</div>',unsafe_allow_html=True)
        st.markdown('<div class="panel-title-row"><span class="panel-title">RW-UCFI Local</span><span class="panel-note">risk-weighted + uncertainty-conditioned</span></div>',unsafe_allow_html=True)
        r=live_xai["local_rw_ucfi"]
        r=r[r["sample_position"]==int(sample_position)].copy()
        if not r.empty:
            r["rw_ucfi_rank"]=r["local_rw_ucfi_rank"]
            render_driver_rows(r,"rw_ucfi_score_entity",topn=10)
        else:
            st.markdown('<div class="notice">RW-UCFI local belum tersedia untuk customer ini karena baris tersebut belum termasuk baris yang dijelaskan.</div>',unsafe_allow_html=True)
    else:
        st.markdown('<div class="notice">Aktifkan opsi SHAP/RW-UCFI live inference untuk menghitung explanation customer baru dengan production model.</div>',unsafe_allow_html=True)
    st.markdown(f'<div class="notice" style="border-left:0;border-right:0;border-bottom:0">Schema coverage: <b>{int(report.get("n_selected_features_available",0)):,}/{int(report.get("n_selected_features_expected",0)):,}</b> feature tersedia. Missing feature akan diimputasi oleh preprocessing pipeline. Untuk keputusan bisnis, gunakan input selengkap mungkin.</div>',unsafe_allow_html=True)


def render_new_customer(paths, data_root, output_root):
    bundle_path=paths["bundle"]
    bundle=None
    n_features=164
    model_name="StackedEnsemble"
    if bundle_path.exists():
        try:
            bundle=cached_bundle(str(bundle_path))
            n_features=len(bundle.get("selected_features",[]))
            model_name=bundle.get("best_model_name","StackedEnsemble")
        except Exception as e:
            st.error(f"Model bundle ditemukan tetapi gagal dibuka: {e}")
    render_top_header("Prediksi Pelanggan Baru", "Masukkan data pelanggan atau unggah berkas untuk klasifikasi risiko instan", "Model", model_name)
    left,right=st.columns([1.22,.99],gap="medium")
    with left:
        st.markdown(f'<div class="panel"><div class="panel-title-row"><span class="panel-title">Unggah Data</span><span class="upload-note">CSV / Excel · {n_features:,} fitur terpilih</span></div></div>',unsafe_allow_html=True)
        if bundle is None:
            st.warning("production_model_bundle.joblib belum tersedia. Jalankan Cell 11 pipeline, lalu copy Stage 5 output.")
            return
        template=build_input_template(bundle,n_rows=1)
        st.download_button("Download template input",data=dataframe_to_excel_bytes(template,"New_Customer_Input"),file_name="new_customer_input_template.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        up=st.file_uploader("Tarik berkas ke sini atau klik untuk memilih",type=["csv","xlsx","xls"],label_visibility="visible",key="new_customer_upload")
        raw_uploaded=None
        if up is not None:
            try:
                raw_uploaded=read_uploaded_table(up)
                st.caption(f"{len(raw_uploaded):,} baris terdeteksi · {len(raw_uploaded.columns):,} kolom")
            except Exception as e:
                st.error(f"File tidak dapat diproses: {e}")

        st.markdown('<div style="padding:.8rem 0 .25rem;font-family:Libre Caslon Text,Georgia,serif;font-size:1.155rem">Opsi Live Explainability</div>',unsafe_allow_html=True)
        xai_enabled=st.checkbox("Aktifkan SHAP/RW-UCFI live inference",value=False,help="Menghitung Kernel SHAP dan RW-UCFI post-hoc pada production model. Prediksi kelas tidak berubah.")
        xai_cfg={"enabled":False}; background_upload=None
        if xai_enabled:
            x1,x2,x3=st.columns(3)
            with x1:
                explain_rows=st.slider("Jumlah baris dijelaskan",1,10,3,1)
            with x2:
                background_size=st.slider("Background SHAP",10,100,30,5)
            with x3:
                top_n=st.slider("Top fitur XAI",5,20,10,1)
            st.markdown('<div style="padding:.55rem 0 .25rem;font-family:Libre Caslon Text,Georgia,serif;font-size:1.100rem">Konfigurasi Bobot RW-UCFI Live</div>',unsafe_allow_html=True)
            w1,w2,w3,w4=st.columns(4)
            with w1: high_w=st.number_input("Bobot High Risk",min_value=0.0,max_value=5.0,value=1.50,step=0.25)
            with w2: med_w=st.number_input("Bobot Medium Risk",min_value=0.0,max_value=5.0,value=1.00,step=0.25)
            with w3: low_w=st.number_input("Bobot Low Risk",min_value=0.0,max_value=5.0,value=0.50,step=0.25)
            with w4: alpha=st.number_input("Alpha Uncertainty",min_value=0.0,max_value=5.0,value=1.00,step=0.25)
            st.caption("Default mengikuti Cell 10 pipeline. Bobot hanya mengubah ranking RW-UCFI/prioritas review; tidak mengubah kelas, probability, atau uncertainty flag.")
            with st.expander("Pengaturan SHAP lanjutan / reference background",expanded=False):
                nsamples=st.number_input("Kernel SHAP nsamples",min_value=50,max_value=1000,value=200,step=50)
                flag_bonus=st.number_input("Uncertain flag bonus",min_value=0.0,max_value=2.0,value=0.25,step=0.05)
                background_upload=st.file_uploader("Upload reference/background dataset (opsional)",type=["csv","xlsx","xls"],key="xai_background_upload")
                if paths.get("xai_background") and paths["xai_background"].exists():
                    st.success(f"Background artifact tersedia: {paths['xai_background'].name}")
                else:
                    ref=discover_reference_dataset(data_root)
                    if ref:
                        st.info(f"Background otomatis dari dataset pipeline: {ref.name}")
                    else:
                        st.warning("Reference background belum ditemukan. Upload dataset referensi atau buat production_shap_background.joblib.")
            xai_cfg={
                "enabled":True,"explain_rows":int(explain_rows),"background_size":int(background_size),"top_n":int(top_n),
                "nsamples":int(nsamples),"risk_weights":{"High Risk":high_w,"Medium Risk":med_w,"Low Risk":low_w},
                "alpha":float(alpha),"flag_bonus":float(flag_bonus)
            }

        if raw_uploaded is not None:
            if st.button("Jalankan Prediksi File",type="primary",use_container_width=True):
                with st.spinner("Menjalankan prediction" + (" + SHAP/RW-UCFI..." if xai_enabled else "...")):
                    out,report,xai,xerr=execute_new_prediction(raw_uploaded,bundle,xai_cfg,paths,data_root,background_upload)
                st.session_state["new_prediction_df"]=out
                st.session_state["new_prediction_report"]=report
                st.session_state["new_xai_results"]=xai
                st.session_state["new_xai_error"]=xerr
                st.session_state["new_prediction_selected_idx"]=0

        st.markdown('<div style="padding:.7rem 0 .2rem;font-size:0.825rem">— atau isi manual / quick entry —</div>',unsafe_allow_html=True)
        sector_options, sector_source, sector_checked = get_sector_options(data_root, paths.get("production_predictions"), output_root)
        with st.form("manual_prediction"):
            name=st.text_input("Nama perusahaan (identitas)",value="")
            c1,c2=st.columns(2)
            with c1:
                if sector_options:
                    sector=st.selectbox(
                        "sectorKey",
                        sector_options,
                        index=0,
                        help=f"Pilihan dibaca dari datasource: {sector_source}"
                    )
                    st.caption(f"Sumber sectorKey: {sector_source}")
                else:
                    sector=st.selectbox(
                        "sectorKey",
                        ["— sectorKey tidak ditemukan di datasource —"],
                        index=0,
                        help="Atur Datasource file / data folder pada sidebar. Folder dashboard tidak harus sama dengan folder pipeline."
                    )
                    if sector_checked:
                        st.caption("Path yang diperiksa: " + " | ".join(dict.fromkeys(sector_checked)))
            with c2:
                revenue=st.number_input(
                    "Revenue", value=None, step=1_000_000.0, placeholder="opsional",
                    help="Satu nilai Revenue dipetakan sekaligus ke totalRevenue dan IS_TotalRevenue."
                )
            c3,c4=st.columns(2)
            with c3:
                cr=st.number_input(
                    "Collection Ratio / CR", value=None, step=0.01, placeholder="opsional",
                    help="Dipetakan ke CR_2024/CR2024. Gunakan skala yang sama dengan database training."
                )
            with c4:
                cyc=st.number_input(
                    "CYC", value=None, step=1.0, placeholder="opsional",
                    help="Dipetakan ke CYC_2024/CYC2024 bila fitur tersebut tersedia pada production bundle."
                )
            c5,c6=st.columns(2)
            with c5:
                ebitda=st.number_input(
                    "EBITDA", value=None, step=1_000_000.0, placeholder="opsional",
                    help="Dipetakan ke fitur EBITDA/ebitda bila tersedia pada production bundle."
                )
            with c6:
                acquisition_age=st.number_input(
                    "Customer Acquisition Age in Year", value=None, step=0.5, placeholder="opsional",
                    help="Dipetakan ke Customer_Acquisition_Age_in_Year."
                )
            submitted=st.form_submit_button("Jalankan Prediksi",type="primary",use_container_width=True)
        if submitted:
            raw={"Company Name":name or "New Customer"}

            # sectorKey comes from the pipeline database dropdown.
            if sector_options and sector:
                for col in feature_targets(bundle,["sectorKey"],"sectorKey"):
                    raw[col]=sector

            # One Revenue input intentionally populates both revenue features requested by the current production schema.
            if revenue is not None:
                for col in feature_targets(bundle,["totalRevenue","TotalRevenue","total_revenue"],"totalRevenue"):
                    raw[col]=revenue
                for col in feature_targets(bundle,["IS_TotalRevenue","IS TotalRevenue","IS Total Revenue","IS_Total_Revenue"],"IS_TotalRevenue"):
                    raw[col]=revenue

            if cr is not None:
                for col in feature_targets(bundle,["CR2024","CR_2024","CR 2024"],"CR_2024"):
                    raw[col]=cr

            if cyc is not None:
                for col in feature_targets(bundle,["CYC2024","CYC_2024","CYC 2024"],"CYC_2024"):
                    raw[col]=cyc

            if ebitda is not None:
                for col in feature_targets(bundle,["EBITDA","ebitda"],"ebitda"):
                    raw[col]=ebitda

            if acquisition_age is not None:
                for col in feature_targets(
                    bundle,
                    ["Customer_Acquisition_Age_in_Year","Customer Acquisition Age in Year","Customer_Acquisition_Age_inYear"],
                    "Customer_Acquisition_Age_in_Year"
                ):
                    raw[col]=acquisition_age

            manual_df=pd.DataFrame([raw])
            with st.spinner("Menjalankan prediction" + (" + SHAP/RW-UCFI..." if xai_enabled else "...")):
                out,report,xai,xerr=execute_new_prediction(manual_df,bundle,xai_cfg,paths,data_root,background_upload)
            st.session_state["new_prediction_df"]=out
            st.session_state["new_prediction_report"]=report
            st.session_state["new_xai_results"]=xai
            st.session_state["new_xai_error"]=xerr
            st.session_state["new_prediction_selected_idx"]=0

    with right:
        out=st.session_state.get("new_prediction_df")
        rep=st.session_state.get("new_prediction_report")
        xai=st.session_state.get("new_xai_results")
        xerr=st.session_state.get("new_xai_error")
        selected_pos=0
        if out is not None and len(out):
            if len(out)>1:
                st.markdown('<div class="output-selector-note">Pilih customer untuk melihat detail hasil prediction</div>',unsafe_allow_html=True)
                positions=list(range(len(out)))
                def _prediction_option_label(pos):
                    r=out.iloc[int(pos)]
                    rc_local=risk_col_name(pd.DataFrame([r]))
                    risk_local=risk_class(r.get(rc_local,"—")) if rc_local else "—"
                    return f"#{int(pos)+1} · {company_name(r)} · {RISK_SHORT.get(risk_local,risk_local)}"
                selected_pos=st.selectbox(
                    "Pilih output customer",
                    options=positions,
                    format_func=_prediction_option_label,
                    key="new_prediction_selected_idx",
                    label_visibility="collapsed",
                )
            else:
                st.session_state["new_prediction_selected_idx"]=0
                selected_pos=0
            render_prediction_result(out.iloc[int(selected_pos)],rep or {},xai,sample_position=int(selected_pos))
            if xerr:
                st.warning(f"Prediction selesai, tetapi live XAI belum berhasil: {xerr}")
            if len(out)>1:
                st.download_button("Download seluruh hasil prediction",data=dataframe_to_excel_bytes(out,"Predictions"),file_name="new_customer_predictions.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
        else:
            st.markdown('<div class="panel"><div class="notice" style="border:0">Hasil prediksi akan tampil di sini setelah file atau quick entry diproses.</div></div>',unsafe_allow_html=True)

    out=st.session_state.get("new_prediction_df")
    xai=st.session_state.get("new_xai_results")
    selected_pos=int(st.session_state.get("new_prediction_selected_idx",0) or 0)
    if out is not None and xai:
        render_live_xai_results(xai,out,paths,top_n=int(xai.get("metadata",{}).get("top_n",10)),selected_position=selected_pos)

def load_optional(path):
    try:
        return load_table(str(path)) if path and path.exists() else None
    except Exception:
        return None


def load_optional_json(path):
    try:
        return load_json(str(path)) if path and path.exists() else {}
    except Exception:
        return {}


def _find_sidebar_logo():
    """Return a local logo path when supplied with the dashboard package."""
    candidates = [
        APP_DIR / "assets" / "logo.png",
        APP_DIR / "assets" / "logo.jpg",
        APP_DIR / "assets" / "logo.jpeg",
        APP_DIR / "assets" / "logo.webp",
        APP_DIR / "logo.png",
        APP_DIR / "logo.jpg",
        APP_DIR / "logo.jpeg",
        APP_DIR / "logo.webp",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


# ------------------------- configuration -------------------------
with st.sidebar:
    sidebar_logo = _find_sidebar_logo()
    if sidebar_logo is not None:
        st.image(str(sidebar_logo), width=165)
    st.markdown('<div class="sidebar-brand">B2B Customer Risk Profile</div>',unsafe_allow_html=True)
    st.markdown('---')
    page=st.radio(
        "Navigation",
        [
            "Portfolio Overview",
            "Portfolio Overview (Observasi)",
            "Existing Customer Data",
            "New Customer Prediction",
        ],
        label_visibility="collapsed",
    )

# Paths are resolved silently. Override with B2B_PIPELINE_ROOT and B2B_DATA_ROOT.
output_root=resolve_output_root(str(DEFAULT_PIPELINE_ROOT))
data_root=resolve_data_source(str(DEFAULT_DATA_ROOT), output_root)
paths=artifact_paths(output_root)
metrics=load_optional_json(paths["final_metrics"])

prod=load_optional(paths["production_predictions"])
final_df=load_optional(paths["final_predictions"])
shap_df=load_optional(paths["shap_local"])
rw_df=load_optional(paths["rw_entity"])
gov_df=load_optional(paths["rw_governance"])

if page.startswith("Portfolio Overview (Observasi)"):
    render_observation_portfolio(final_df, prod, metrics, paths, data_root=data_root)
elif page.startswith("Portfolio Overview"):
    render_portfolio(prod,gov_df,metrics,paths["production_predictions"],data_root=data_root)
elif page.startswith("Existing"):
    render_existing_customer(prod,final_df,shap_df,rw_df,gov_df,data_root=data_root,tensor_path=paths.get("rw_tensor"))
else:
    render_new_customer(paths, data_root, output_root)
