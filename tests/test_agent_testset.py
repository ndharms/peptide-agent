import re
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List

import pandas as pd
import typer

# Reuse the agent's batch runner and configuration knobs
from src.pipeline.agent import run_agent_batch, TOP_K_DEFAULT, LLM_MODEL_DEFAULT


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


INTERVAL_PATTERN = r"\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"
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
    labeled_patterns = {
        "PH": r"PH\s*[:=-]\s*(" + INTERVAL_PATTERN + r")",
        "CONCENTRATION_LOG_MGML": r"Concentration.*?\b(" + INTERVAL_PATTERN + r")",
        "TIME_MINUTES": r"Estimated\s*Time.*?\b(" + INTERVAL_PATTERN + r"|\(<\d+\)|\(>\d+\))",
        # Temperature: allow interval or single number
        "TEMPERATURE_C_INTERVAL": r"Temperature.*?\b(" + INTERVAL_PATTERN + r")",
        "TEMPERATURE_C_NUMBER": r"Temperature.*?(-?\d+(?:\.\d+)?)",
        "SOLVENT": r"Solvent\s*[:=-]\s*([A-Za-z0-9/ \-\+]+)",
    }

    # PH
    m = re.search(labeled_patterns["PH"], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        result["PH"] = m.group(1)

    # Concentration
    m = re.search(labeled_patterns["CONCENTRATION_LOG_MGML"], text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        result["CONCENTRATION_LOG_MGML"] = m.group(1)

    # Time
    m = re.search(labeled_patterns["TIME_MINUTES"], text, flags=re.IGNORECASE | re.DOTALL)
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
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        result["TEMPERATURE_C"] = m.group(1)
    else:
        m = re.search(
            labeled_patterns["TEMPERATURE_C_NUMBER"],
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m:
            result["TEMPERATURE_C"] = m.group(1)

    # Solvent
    m = re.search(labeled_patterns["SOLVENT"], text, flags=re.IGNORECASE)
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


def parse_interval(interval: str) -> Optional[Tuple[float, float]]:
    m = re.match(r"\(\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)", interval.strip())
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def value_in_interval(value: float, interval: str) -> bool:
    bounds = parse_interval(interval)
    if not bounds:
        return False
    lo, hi = bounds
    return value >= lo and value <= hi


# ---------------------------
# Scoring
# ---------------------------


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
    if pred_ph and gt_ph is not None:
        ph_ok = value_in_interval(gt_ph, pred_ph)

    # Concentration interval membership (pred is LOG_MGML bucket, gt converted to log bucket)
    pred_conc = pred.get("CONCENTRATION_LOG_MGML")
    if pred_conc and gt_conc_bucket is not None:
        conc_ok = pred_conc.strip() == gt_conc_bucket

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

    # Solvent: normalize and compare
    pred_solvent = normalize_solvent(pred.get("SOLVENT"))
    if pred_solvent and gt_solvent:
        solv_ok = pred_solvent == gt_solvent

    # Time minutes interval membership
    pred_time = pred.get("TIME_MINUTES")
    if pred_time and gt_time_min is not None:
        time_ok = value_in_interval(gt_time_min, pred_time)  # allows open intervals as encoded

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


@app.command()
def evaluate(
    csv_path: Path = typer.Option(Path("data/test_split.csv"), help="Path to test CSV"),
    llm_model: str = typer.Option(LLM_MODEL_DEFAULT, help="LLM model id"),
    top_k: int = typer.Option(TOP_K_DEFAULT, help="Retriever top-k"),
    limit: Optional[int] = typer.Option(None, help="Limit number of rows"),
    save_jsonl: Optional[Path] = typer.Option(None, help="Path to save results JSONL"),
    batch_size: int = typer.Option(20, help="Batch size for LLM calls"),
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

            row_result = {
                "row_index": int(idx),
                "peptide_code": peptide_code,
                "morphology": morphology,
                "pred": pred,
                "scores": scores,
                "report": report,
            }
            results.append(row_result)

            typer.echo(
                f"[{idx}] total={scores['total']} "
                f"ph={scores['ph_ok']} conc={scores['conc_ok']} "
                f"temp={scores['temp_ok']} solv={scores['solv_ok']} "
                f"time={scores['time_ok']}"
            )

    avg_score = total_score / max(len(df), 1)
    typer.echo(f"Evaluated {len(df)} rows")
    typer.echo(f"Average score: {avg_score:.2f} out of 5")

    if save_jsonl:
        save_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with open(save_jsonl, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    app()
