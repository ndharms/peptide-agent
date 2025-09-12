import re
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import typer

# Ensure 'src' is importable when running this file directly.
# Example: python tests/test_agent_testset.py evaluate ...
import os
import sys
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Reuse the agent's batch runner and configuration knobs
from src.pipeline.agent import run_agent_batch, TOP_K_DEFAULT, LLM_MODEL_DEFAULT  # noqa: E402


app = typer.Typer(help="Evaluate the peptide synthesis agent against test_split.csv")


# ---------------------------
# Parsing and normalization
# ---------------------------

PH_BUCKETS: List[Tuple[float, float]] = [
    (1.0, 3.0),
    (3.0, 5.0),
    (5.0, 6.5),
    (6.5, 7.5),
    (7.5, 9.0),
    (9.0, 11.0),
    (11.0, 14.0),
]
PH_BUCKET_STRINGS = ["(1,3)", "(3,5)", "(5,6.5)", "(6.5,7.5)", "(7.5,9)", "(9,11)", "(11,14)"]

CONC_LOG_BUCKETS: List[Tuple[float, float]] = [
    (-3.0, -1.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
]
CONC_LOG_BUCKET_STRINGS = ["(-3,-1)", "(-1,0)", "(0,1)", "(1,2)", "(2,3)"]

TIME_MIN_BUCKETS: List[Tuple[Optional[float], Optional[float]]] = [
    (None, 30.0),  # (<30)
    (30.0, 60.0),  # (30,60)
    (60.0, 120.0),  # (60,120)
    (120.0, 240.0),  # (120,240)
    (240.0, None),  # (>240)
]
TIME_MIN_BUCKET_STRINGS = ["(<30)", "(30,60)", "(60,120)", "(120,240)", "(>240)"]

TEMPERATURE_TOLERANCE = 2.5  # degrees C tolerance when model outputs a single number


def _to_float(value: Any) -> Optional[float]:
    """Parse a float from various CSV cell formats; support ranges like '6–7' by averaging."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "nan":
        return None
    # Replace unicode en dash or em dash with '-'
    s = s.replace("–", "-").replace("—", "-")
    # If it's a range like '6-7' or '4-10', average the endpoints
    if "-" in s and not s.startswith("-"):
        parts = [p.strip() for p in s.split("-") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass
        if len(nums) >= 2:
            return sum(nums[:2]) / 2.0
    # Otherwise, parse as single float
    try:
        return float(s)
    except Exception:
        return None


def ph_to_bucket(ph: Optional[float]) -> Optional[str]:
    if ph is None:
        return None
    for (lo, hi), label in zip(PH_BUCKETS, PH_BUCKET_STRINGS):
        if ph >= lo and ph <= hi:
            return label
    return None


def log10_safe(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    if x <= 0:
        return None
    try:
        return math.log10(x)
    except Exception:
        return None


def conc_log_to_bucket(log_val: Optional[float]) -> Optional[str]:
    if log_val is None:
        return None
    for (lo, hi), label in zip(CONC_LOG_BUCKETS, CONC_LOG_BUCKET_STRINGS):
        if log_val >= lo and log_val <= hi:
            return label
    return None


def time_to_bucket_minutes(minutes: Optional[float]) -> Optional[str]:
    if minutes is None:
        return None
    for (lo, hi), label in zip(TIME_MIN_BUCKETS, TIME_MIN_BUCKET_STRINGS):
        if lo is None and minutes < hi:
            return label
        if hi is None and minutes > lo:
            return label
        if lo is not None and hi is not None and minutes >= lo and minutes <= hi:
            return label
    return None


SOLVENT_ALIASES = {
    "h2o": "water",
    "water": "water",
    "pb": "pbs",
    "pbs": "pbs",
    "phosphate buffer": "pbs",
    "phosphate-buffered saline": "pbs",
    "acn": "acetonitrile",
    "acetonitrile": "acetonitrile",
    "dcm": "dichloromethane",
    "dichloromethane": "dichloromethane",
    "dmf": "dimethylformamide",
    "dimethylformamide": "dimethylformamide",
    "nmp": "n-methyl-2-pyrrolidone",
    "n-methyl-2-pyrrolidone": "n-methyl-2-pyrrolidone",
    "chloroform": "chloroform",
    "methanol": "methanol",
    "ethanol": "ethanol",
    "toluene": "toluene",
    "hexane": "hexane",
    "dmsO": "dmso",
    "dmso": "dmso",
    "thf": "thf",
    "hfp": "hfp",
    "hepes": "hepes",
    "hepes/h2o": "hepes",
    "hepes buffer": "hepes",
    "hepse": "hepes",
    "hepes ": "hepes",
    "hepes  ": "hepes",
    "hepes\t": "hepes",
    "hepes\n": "hepes",
    "hepes/h2o": "hepes",
    "hepes/water": "hepes",
    "hepes ": "hepes",
    "hepes  ": "hepes",
    "hepes\t": "hepes",
    "hepes\n": "hepes",
    "hepes/h2o": "hepes",
    "hepes/water": "hepes",
    "hepes ": "hepes",
    "trис": "tris",  # in case of unicode anomalies
    "tris": "tris",
    "acetone": "acetone",
    "ethyl acetate": "ethyl acetate",
    "chlorobenzene": "chlorobenzene",
    "tfa": "trifluoroacetic acid",
    "trifluoroacetic acid": "trifluoroacetic acid",
    "pb/h2o": "pbs",
    "pbs/h2o": "pbs",
    "hfp/h2o": "hfp/h2o",
}


def normalize_solvent(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    base = s.strip().lower()
    # Remove parentheses content and extra punctuation for robust matching
    base = re.sub(r"\([^)]*\)", "", base).strip()
    base = base.replace(",", " ").replace("  ", " ")
    if base in SOLVENT_ALIASES:
        return SOLVENT_ALIASES[base]
    return base


INTERVAL_PATTERN = r"[\(\[]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\)\]]"
SIMPLE_BUCKETS = set(PH_BUCKET_STRINGS + CONC_LOG_BUCKET_STRINGS + TIME_MIN_BUCKET_STRINGS)


def parse_agent_report(report: str) -> Dict[str, Optional[str]]:
    """
    Heuristic parser for the agent's text report.
    Returns dictionary with keys:
      - PH (interval string)
      - CONCENTRATION_LOG_MGML (interval string)
      - TEMPERATURE_C (either number string or interval string)
      - SOLVENT (string)
      - TIME_MINUTES (interval string)
    """
    result: Dict[str, Optional[str]] = {
        "PH": None,
        "CONCENTRATION_LOG_MGML": None,
        "TEMPERATURE_C": None,
        "SOLVENT": None,
        "TIME_MINUTES": None,
    }
    text = report

    # Try to find explicit labeled lines first
    # Strict line-anchored patterns to match the fixed 5-line report format
    labeled_patterns = {
        "PH": r"^\s*PH\s*:\s*(" + INTERVAL_PATTERN + r")",
        "CONCENTRATION_LOG_MGML": (
            r"^\s*Concentration\s*\(log M\)\s*:\s*(" + INTERVAL_PATTERN + r")"
        ),
        "TIME_MINUTES": r"^\s*Estimated\s*Time\s*\(minutes\)\s*:\s*(" + INTERVAL_PATTERN + r")",
        "TEMPERATURE_C_INTERVAL": r"^\s*Temperature\s*\(C\)\s*:\s*(" + INTERVAL_PATTERN + r")",
        "TEMPERATURE_C_NUMBER": r"^\s*Temperature\s*\(C\)\s*:\s*(-?\d+(?:\.\d+)?)",
        "SOLVENT": r"^\s*Solvent\s*:\s*([A-Za-z0-9/ \-\+]+)",
    }

    # PH
    m = re.search(labeled_patterns["PH"], text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        result["PH"] = m.group(1)

    # Concentration
    m = re.search(
        labeled_patterns["CONCENTRATION_LOG_MGML"],
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        result["CONCENTRATION_LOG_MGML"] = m.group(1)

    # Time
    m = re.search(
        labeled_patterns["TIME_MINUTES"],
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        candidate = m.group(1)
        # Normalize forms like (<30) or (30,60) or (>240)
        m2 = re.search(INTERVAL_PATTERN, candidate)
        if m2:
            result["TIME_MINUTES"] = m2.group(0)
        elif candidate.startswith("(<"):
            result["TIME_MINUTES"] = "(<{})".format(re.sub(r"[^\d]", "", candidate))
        elif candidate.startswith("(>"):
            result["TIME_MINUTES"] = "(>{})".format(re.sub(r"[^\d]", "", candidate))

    # Temperature
    m = re.search(
        labeled_patterns["TEMPERATURE_C_INTERVAL"],
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if m:
        result["TEMPERATURE_C"] = m.group(1)
    else:
        m = re.search(
            labeled_patterns["TEMPERATURE_C_NUMBER"],
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m:
            result["TEMPERATURE_C"] = m.group(1)

    # Solvent
    m = re.search(labeled_patterns["SOLVENT"], text, flags=re.IGNORECASE | re.MULTILINE)
    if m:
        result["SOLVENT"] = m.group(1).strip()

    # Fallback: scan for any known bucket strings if labels were missed
    if result["PH"] is None:
        for b in PH_BUCKET_STRINGS:
            if b in text:
                result["PH"] = b
                break
    if result["CONCENTRATION_LOG_MGML"] is None:
        for b in CONC_LOG_BUCKET_STRINGS:
            if b in text:
                result["CONCENTRATION_LOG_MGML"] = b
                break
    if result["TIME_MINUTES"] is None:
        for b in TIME_MIN_BUCKET_STRINGS:
            if b in text:
                result["TIME_MINUTES"] = b
                break

    return result


def parse_interval(interval: str) -> Optional[Tuple[float, float, bool, bool]]:
    s = interval.strip()
    m = re.match(
        r"([\(\[])\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*([\)\]])",
        s,
    )
    if not m:
        return None
    lo = float(m.group(2))
    hi = float(m.group(3))
    lo_incl = m.group(1) == "["
    hi_incl = m.group(4) == "]"
    return lo, hi, lo_incl, hi_incl


def value_in_interval(value: float, interval: str) -> bool:
    bounds = parse_interval(interval)
    if not bounds:
        return False
    lo, hi, lo_incl, hi_incl = bounds
    if lo_incl:
        if value < lo:
            return False
    else:
        if value <= lo:
            return False
    if hi_incl:
        if value > hi:
            return False
    else:
        if value >= hi:
            return False
    return True


# ---------------------------
# Scoring
# ---------------------------

# --- Soft scoring helpers ---

def _interval_center_width(interval: str) -> Optional[Tuple[float, float]]:
    """Return (center, half_width) for '(a,b)' / '[a,b]' strings."""
    bounds = parse_interval(interval)
    if not bounds:
        return None
    lo, hi, _, _ = bounds
    c = 0.5 * (lo + hi)
    w = 0.5 * abs(hi - lo)
    return c, w

def _clamped_linear(score: float) -> float:
    return max(0.0, min(1.0, score))

def score_ph_soft(pred_interval: Optional[str], gt_ph: Optional[float]) -> float:
    """
    1.0 if GT inside pred interval.
    If outside, linearly decay with distance normalized by the full pH scale (0–14) or by bucket size.
    """
    if gt_ph is None or pred_interval is None:
        return 1.0  # no ground truth → neutral credit
    if value_in_interval(gt_ph, pred_interval):
        return 1.0
    cw = _interval_center_width(pred_interval)
    if not cw:
        return 0.0
    c, w = cw
    # Normalize distance by a reasonable scale; use half_width + 1.0 as softness
    dist = abs(gt_ph - c)
    scale = max(w, 0.5) + 1.0  # e.g., 1.5–2.0 typical
    return _clamped_linear(1.0 - dist / (3.0 * scale))  # gentle tail-off

def score_temp_soft(pred_temp: Optional[str], gt_temp: Optional[float], tol: float = TEMPERATURE_TOLERANCE) -> float:
    """
    If interval: 1.0 inside; outside decays with distance normalized by 2*tol.
    If number: 1.0 at 0 diff, drops to ~0 at ~4*tol.
    """
    if gt_temp is None or pred_temp is None:
        return 1.0
    pred_temp = str(pred_temp).strip()
    if re.match(INTERVAL_PATTERN, pred_temp):
        if value_in_interval(gt_temp, pred_temp):
            return 1.0
        cw = _interval_center_width(pred_temp)
        if not cw:
            return 0.0
        c, w = cw
        dist = abs(gt_temp - c)
        scale = max(w, tol)
        return _clamped_linear(1.0 - dist / (3.0 * scale))
    # numeric
    try:
        pv = float(pred_temp)
    except Exception:
        return 0.0
    diff = abs(pv - gt_temp)
    # Smooth decay: 1 at diff=0; ~0.5 at diff=tol; → 0 near ~4*tol
    return _clamped_linear(1.0 - diff / (4.0 * tol))

def _bucket_index(b: str, all_buckets: List[str]) -> Optional[int]:
    try:
        return all_buckets.index(b)
    except ValueError:
        return None

def score_conc_soft(pred_bucket: Optional[str], gt_log_bucket: Optional[str]) -> float:
    """
    Full credit for same bucket, partial for neighbors, small credit for 2 buckets away.
    """
    if gt_log_bucket is None or pred_bucket is None:
        return 1.0
    pi = _bucket_index(pred_bucket, CONC_LOG_BUCKET_STRINGS)
    gi = _bucket_index(gt_log_bucket, CONC_LOG_BUCKET_STRINGS)
    if pi is None or gi is None:
        return 0.0
    d = abs(pi - gi)
    if d == 0:
        return 1.0
    if d == 1:
        return 0.7
    if d == 2:
        return 0.4
    return 0.0

# --- Solvent similarity via chemotype classes + light string match ---

# Map various normalized solvents into broader classes for similarity
SOLVENT_CLASS = {
    "water": "aqueous",
    "pbs": "aqueous",
    "tris": "aqueous",
    "hepes": "aqueous",
    "dmso": "polar_aprotic",
    "acetonitrile": "polar_aprotic",
    "dichloromethane": "nonpolar_org",
    "dimethylformamide": "polar_aprotic",
    "n-methyl-2-pyrrolidone": "polar_aprotic",
    "trifluoroacetic acid": "acidic",
    "thf": "polar_aprotic",
    "methanol": "polar_protic",
    "ethanol": "polar_protic",
    "toluene": "nonpolar_org",
    "acetone": "polar_aprotic",
    "chlorobenzene": "nonpolar_org",
}

def _string_similarity(a: str, b: str) -> float:
    # Cheap token overlap similarity (0..1)
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union

def score_solvent_soft(pred_solvent: Optional[str], gt_solvent_raw: Optional[str]) -> float:
    """
    1.0 if exact normalized match.
    0.8 if same solvent class (aqueous, polar_aprotic, etc.).
    0.5 if high token similarity (e.g., 'water buffer' vs 'pbs').
    else 0.
    """
    if not gt_solvent_raw:
        return 1.0
    gt_norm = normalize_solvent(gt_solvent_raw) or ""
    pred_norm = normalize_solvent(pred_solvent or "") or ""
    if not pred_norm:
        return 0.0
    if pred_norm == gt_norm:
        return 1.0
    cls_pred = SOLVENT_CLASS.get(pred_norm, "")
    cls_gt = SOLVENT_CLASS.get(gt_norm, "")
    if cls_pred and cls_pred == cls_gt:
        return 0.8
    if _string_similarity(pred_norm, gt_norm) >= 0.5:
        return 0.5
    return 0.0

def score_time_soft(pred_interval: Optional[str], gt_minutes: Optional[float]) -> float:
    """
    1.0 if GT inside interval, partial if near the center, else decay by distance.
    """
    if gt_minutes is None or pred_interval is None:
        return 1.0
    if value_in_interval(gt_minutes, pred_interval):
        return 1.0
    cw = _interval_center_width(pred_interval)
    if not cw:
        return 0.0
    c, w = cw
    dist = abs(gt_minutes - c)
    scale = max(w, 15.0)  # at least ~15 min softness
    return _clamped_linear(1.0 - dist / (3.0 * scale))


def score_row_soft(pred: Dict[str, Optional[str]], row: pd.Series,
                   weights: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Continuous scores 0..1 per field + weighted total 0..5 (default equal weights).
    Fields: PH, CONC_LOG_MGML, TEMP, SOLVENT, TIME
    """
    if weights is None:
        weights = {"PH":1.0, "CONC":1.0, "TEMP":1.0, "SOLV":1.0, "TIME":1.0}

    # Ground truth normalization (reuse your helpers)
    gt_ph = _to_float(row.get("PH"))
    gt_conc_mgml = _to_float(row.get("CONCENTRATION_mg ml") or row.get("CONCENTRATION_mgml"))
    gt_conc_log = log10_safe(gt_conc_mgml)
    gt_conc_bucket = conc_log_to_bucket(gt_conc_log)
    gt_temp = _to_float(row.get("TEMPERATURE_C"))
    gt_solvent_raw = str(row.get("SOLVENT") or "").strip()
    gt_time_min = _to_float(row.get("Time (min)") or row.get("Time(min)") or row.get("Time"))

    # Predictions
    s_ph = score_ph_soft(pred.get("PH"), gt_ph)
    s_conc = score_conc_soft(pred.get("CONCENTRATION_LOG_MGML"), gt_conc_bucket)
    s_temp = score_temp_soft(pred.get("TEMPERATURE_C"), gt_temp)
    s_solv = score_solvent_soft(pred.get("SOLVENT"), gt_solvent_raw)
    s_time = score_time_soft(pred.get("TIME_MINUTES"), gt_time_min)

    # Weighted total scaled to 0..5 to align with your original scale
    wsum = sum(weights.values()) or 1.0
    total_soft = 5.0 * (weights["PH"]*s_ph + weights["CONC"]*s_conc + weights["TEMP"]*s_temp +
                        weights["SOLV"]*s_solv + weights["TIME"]*s_time) / wsum

    return {
        "ph": round(s_ph, 4),
        "conc": round(s_conc, 4),
        "temp": round(s_temp, 4),
        "solv": round(s_solv, 4),
        "time": round(s_time, 4),
        "total_soft": round(total_soft, 4),
    }




def score_row(pred: Dict[str, Optional[str]], row: pd.Series) -> Dict[str, Any]:
    """
    Score a single row comparing predicted intervals to ground truth values.
    Returns a dict with per-field booleans and total score (0..5).
    Fields: PH, CONC_LOG_MGML, TEMPERATURE_C, SOLVENT, TIME_MINUTES
    """
    # Ground truths
    gt_ph = _to_float(row.get("PH"))

    gt_conc_mgml = _to_float(row.get("CONCENTRATION_mg ml") or row.get("CONCENTRATION_mgml"))
    gt_conc_log = log10_safe(gt_conc_mgml)
    gt_conc_bucket = conc_log_to_bucket(gt_conc_log)

    gt_temp = _to_float(row.get("TEMPERATURE_C"))

    gt_solvent_raw = str(row.get("SOLVENT") or "").strip()
    gt_solvent = normalize_solvent(gt_solvent_raw)

    gt_time_min = _to_float(row.get("Time (min)") or row.get("Time(min)") or row.get("Time"))

    # Predictions
    ph_ok = False
    conc_ok = False
    temp_ok = False
    solv_ok = False
    time_ok = False

    # PH interval membership
    pred_ph = pred.get("PH")
    # debug removed
    if pred_ph and gt_ph is not None:
        ph_ok = value_in_interval(gt_ph, pred_ph)
    if gt_ph is None:
        ph_ok = True

    # Concentration interval membership (pred is LOG_MGML bucket, gt converted to log bucket)
    pred_conc = pred.get("CONCENTRATION_LOG_MGML")
    if pred_conc and gt_conc_bucket is not None:
        conc_ok = pred_conc.strip() == gt_conc_bucket
    if gt_conc_bucket is None:
        conc_ok = True

    # Temperature: allow predicted interval or single number; compare with tolerance for number
    pred_temp = pred.get("TEMPERATURE_C")
    if pred_temp and gt_temp is not None:
        if re.match(INTERVAL_PATTERN, pred_temp.strip()):
            temp_ok = value_in_interval(gt_temp, pred_temp)
        else:
            try:
                pred_temp_val = float(str(pred_temp).strip())
                temp_ok = abs(pred_temp_val - gt_temp) <= TEMPERATURE_TOLERANCE
            except Exception:
                temp_ok = False

    if gt_temp is None:
        temp_ok = True

    # Solvent: normalize and compare
    pred_solvent = normalize_solvent(pred.get("SOLVENT"))
    if pred_solvent and gt_solvent:
        solv_ok = pred_solvent == gt_solvent
    if not gt_solvent:
        solv_ok = True

    # Time minutes interval membership
    pred_time = pred.get("TIME_MINUTES")
    if pred_time and gt_time_min is not None:
        time_ok = value_in_interval(gt_time_min, pred_time)
    if gt_time_min is None:
        time_ok = True

    total = int(ph_ok) + int(conc_ok) + int(temp_ok) + int(solv_ok) + int(time_ok)

    return {
        "ph_ok": ph_ok,
        "conc_ok": conc_ok,
        "temp_ok": temp_ok,
        "solv_ok": solv_ok,
        "time_ok": time_ok,
        "total": total,
    }


# ---------------------------
# Runner
# ---------------------------


@app.command(name="evaluate")
def evaluate(
    csv_path: Path = typer.Option(
        Path("data/test_split.csv"),
        "--csv-path",
        help="Path to test CSV",
    ),
    llm_model: str = typer.Option(
        LLM_MODEL_DEFAULT, "--llm-model", help="LLM model id"
    ),
    top_k: int = typer.Option(TOP_K_DEFAULT, "--top-k", help="Retriever top-k"),
    limit: Optional[int] = typer.Option(None, "--limit", help="Limit number of rows"),
    save_jsonl: Optional[Path] = typer.Option(
        None,
        "--save-jsonl",
        help="Path to save results JSONL",
    ),
    save_csv: Optional[Path] = typer.Option(
        None,
        "--save-csv",
        help="Path to save per-row results CSV",
    ),
    batch_size: int = typer.Option(20, "--batch-size", help="Batch size for LLM calls"),
):
    """
    For each row in test_split.csv:
      - Call the agent with (peptide_code, morphology)
      - Parse predicted intervals from the agent's report
      - Score 1 for each of 5 conditions if GT value falls within predicted interval (or matches)
      - Report the average score across all rows (0..5)
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)

    results: List[Dict[str, Any]] = []
    total_score = 0
    # Per-parameter aggregates
    ph_sum = 0
    conc_sum = 0
    temp_sum = 0
    solv_sum = 0
    time_sum = 0

    # do the same for soft 
    soft_total = 0.0
    soft_ph_sum = 0.0
    soft_conc_sum = 0.0
    soft_temp_sum = 0.0
    soft_solv_sum = 0.0
    soft_time_sum = 0.0
    # ---------------------------    
    # Rows for CSV export
    csv_rows: List[Dict[str, Any]] = []

    # Build request payloads and keep row metadata for scoring
    requests: List[Dict[str, str]] = []
    rows_meta: List[Tuple[int, pd.Series, str, str]] = []

    # Resolve peptide_code and morphology for each row first
    code_candidates = [
        "PEPTIDE_CODE",
        "Peptide Code",
        "Peptide",
        "PEPTIDE",
        "Sequence",
        "PEPTIDE CODE",
        "PEPTIDE_code",
        "PEPTIDE CODE ",
    ]
    morph_candidates = [
        "Morphology",
        "MORPHOLOGY",
        "Target",
        "TARGET_STRUCTURAL_ASSEMBLY",
    ]

    for idx, row in df.iterrows():
        peptide_code = ""
        for col in code_candidates:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    peptide_code = str(val).strip()
                    break
        if not peptide_code:
            peptide_code = str(row.iloc[0]).strip()

        morphology = ""
        for col in morph_candidates:
            if col in df.columns:
                val = row.get(col)
                if pd.notna(val) and str(val).strip():
                    morphology = str(val).strip()
                    break
        if not morphology:
            morphology = "none"

        requests.append(
            {
                "peptide_code": peptide_code,
                "target_structural_assembly": morphology,
            }
        )
        rows_meta.append((int(idx), row, peptide_code, morphology))

    # Process in batches to reduce API calls
    for start in range(0, len(requests), batch_size):
        end = min(start + batch_size, len(requests))
        chunk_reqs = requests[start:end]
        chunk_meta = rows_meta[start:end]

        reports = run_agent_batch(
            requests=chunk_reqs,
            top_k=top_k,
            llm_model=llm_model,
            refresh_index=False,  # reuse cached FAISS index
        )

        # Parse, score, and collect results
        for (idx, row, peptide_code, morphology), report in zip(chunk_meta, reports):
            pred = parse_agent_report(report or "")
            scores = score_row(pred, row)
            total_score += scores["total"]

        
            # keep your hard score_total (0..5) and also compute the soft (0..5)
            soft = score_row_soft(pred, row)


            row_result = {
                "row_index": int(idx),
                "peptide_code": peptide_code,
                "morphology": morphology,
                "pred": pred,
                "scores": scores,
                "soft_scores": soft,
                "report": report,
            }
            results.append(row_result)

            # Aggregate per-parameter sums
            ph_sum += int(scores["ph_ok"])
            conc_sum += int(scores["conc_ok"])
            temp_sum += int(scores["temp_ok"])
            solv_sum += int(scores["solv_ok"])
            time_sum += int(scores["time_ok"])

            # do the same for soft scores
            soft_ph_sum += soft["ph"]
            soft_conc_sum += soft["conc"]
            soft_temp_sum += soft["temp"]
            soft_solv_sum += soft["solv"]
            soft_time_sum += soft["time"]
            soft_total += soft["total_soft"]
            # ---------------------------
            

            # Ground-truth normalized values for CSV
            gt_ph_val = _to_float(row.get("PH"))
            gt_conc_mgml = _to_float(
                row.get("CONCENTRATION_mg ml") or row.get("CONCENTRATION_mgml")
            )
            gt_conc_log = log10_safe(gt_conc_mgml)
            gt_conc_bucket = conc_log_to_bucket(gt_conc_log)
            gt_temp_val = _to_float(row.get("TEMPERATURE_C"))
            gt_solvent_raw = str(row.get("SOLVENT") or "").strip()
            gt_solvent_norm = normalize_solvent(gt_solvent_raw)
            gt_time_min = _to_float(
                row.get("Time (min)") or row.get("Time(min)") or row.get("Time")
            )

            # Append CSV row
            csv_rows.append(
                {
                    "row_index": int(idx),
                    "peptide_code": peptide_code,
                    "morphology": morphology,
                    "gt_PH": gt_ph_val,
                    "gt_CONCENTRATION_mgml": gt_conc_mgml,
                    "gt_CONC_LOG": gt_conc_log,
                    "gt_CONC_BUCKET": gt_conc_bucket,
                    "gt_TEMPERATURE_C": gt_temp_val,
                    "gt_SOLVENT_raw": gt_solvent_raw,
                    "gt_SOLVENT_norm": gt_solvent_norm,
                    "gt_TIME_MIN": gt_time_min,
                    "pred_PH": pred.get("PH"),
                    "pred_CONCENTRATION_LOG_MGML": pred.get("CONCENTRATION_LOG_MGML"),
                    "pred_TEMPERATURE_C": pred.get("TEMPERATURE_C"),
                    "pred_SOLVENT": pred.get("SOLVENT"),
                    "pred_TIME_MINUTES": pred.get("TIME_MINUTES"),
                    "score_PH": int(scores["ph_ok"]),
                    "score_CONC": int(scores["conc_ok"]),
                    "score_TEMP": int(scores["temp_ok"]),
                    "score_SOLV": int(scores["solv_ok"]),
                    "score_TIME": int(scores["time_ok"]),
                    "score_TOTAL": int(scores["total"]),
                    "report": report,
                    "soft_score_PH": float(soft["ph"]),
                    "soft_score_CONC": float(soft["conc"]),
                    "soft_score_TEMP": float(soft["temp"]),
                    "soft_score_SOLV": float(soft["solv"]),
                    "soft_score_TIME": float(soft["time"]),
                    "soft_score_TOTAL": float(soft["total_soft"]),
                }
            )

            typer.echo(
                f"[{idx}] total={scores['total']} "
                f"ph={scores['ph_ok']} conc={scores['conc_ok']} "
                f"temp={scores['temp_ok']} solv={scores['solv_ok']} "
                f"time={scores['time_ok']}"
            )
            typer.echo(f"     soft_total={soft['total_soft']:.2f} "
                          f"ph={soft['ph']:.2f} conc={soft['conc']:.2f} "
                            f"temp={soft['temp']:.2f} solv={soft['solv']:.2f} "
                            f"time={soft['time']:.2f}")
            typer.echo(f"     Pred: {pred}")

    n = max(len(df), 1)
    avg_score = total_score / n

    # Per-parameter averages
    avg_ph = ph_sum / n
    avg_conc = conc_sum / n
    avg_temp = temp_sum / n
    avg_solv = solv_sum / n
    avg_time = time_sum / n

    # Soft averages
    avg_soft_total = soft_total / n
    avg_soft_ph = soft_ph_sum / n
    avg_soft_conc = soft_conc_sum / n
    avg_soft_temp = soft_temp_sum / n
    avg_soft_solv = soft_solv_sum / n
    avg_soft_time = soft_time_sum / n
    typer.echo("-----")
    typer.echo(f"Average soft total score: {avg_soft_total:.2f} out of 5")

    typer.echo(f"Evaluated {len(df)} rows")
    typer.echo(f"Average total score: {avg_score:.2f} out of 5")
    typer.echo(
        "Averages — "
        f"ph: {avg_ph:.2f}, conc: {avg_conc:.2f}, temp: {avg_temp:.2f}, "
        f"solv: {avg_solv:.2f}, time: {avg_time:.2f}"
    )
    typer.echo(
        "Soft averages — "
        f"ph: {avg_soft_ph:.2f}, conc: {avg_soft_conc:.2f}, temp: {avg_soft_temp:.2f}, "
        f"solv: {avg_soft_solv:.2f}, time: {avg_soft_time:.2f}"
    )

    if save_jsonl:
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(save_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    if save_csv:
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(csv_rows)
        df_out.to_csv(save_csv, index=False)
        typer.echo(f"Saved CSV to {save_csv}")


if __name__ == "__main__":
    app()
