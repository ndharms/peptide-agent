#!/usr/bin/env python3
"""Evaluation script for peptide agent predictions on test data."""

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from peptide_agent.config import Settings
from peptide_agent.runner.main import predict_batch
from peptide_agent.schemas import PeptideSynthesisConditions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_test_data(csv_path: Path) -> pd.DataFrame:
    """Load test data from CSV file."""
    logger.info(f"Loading test data from {csv_path}")
    df = pd.read_csv(csv_path)
    # Remove empty rows
    df = df.dropna(subset=["PEPTIDE_CODE"])
    logger.info(f"Loaded {len(df)} test examples")
    return df


def convert_to_batch_requests(df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert dataframe to predict_batch input format."""
    requests = []
    for _, row in df.iterrows():
        requests.append(
            {
                "peptide_code": str(row["PEPTIDE_CODE"]),
                "target_structural_assembly": str(row["Morphology"]),
            }
        )
    logger.info(f"Created {len(requests)} batch requests")
    return requests


def mg_ml_to_log_m(mg_ml: float, peptide_code: str) -> float:
    """Convert mg/ml concentration to log M.
    
    Uses approximate molecular weight based on amino acid count.
    Average amino acid MW ~ 110 Da.
    """
    if pd.isna(mg_ml) or mg_ml <= 0:
        return float('nan')
    
    # Estimate molecular weight (rough approximation)
    aa_count = len(peptide_code)
    mw_da = aa_count * 110  # Da (g/mol)
    
    # Convert mg/ml to M
    # mg/ml to g/L: multiply by 1
    # g/L to mol/L: divide by MW
    concentration_m = (mg_ml * 1.0) / mw_da
    
    # Return log M
    return math.log10(concentration_m)


def value_in_interval(value: float, lower: float, upper: float, 
                      lower_inclusive: bool, upper_inclusive: bool) -> bool:
    """Check if value falls within interval bounds."""
    if pd.isna(value):
        return False
    
    # Convert to float to handle any string values
    try:
        value = float(value)
        lower = float(lower)
        upper = float(upper)
    except (ValueError, TypeError):
        return False
    
    lower_ok = (value >= lower) if lower_inclusive else (value > lower)
    upper_ok = (value <= upper) if upper_inclusive else (value < upper)
    
    return lower_ok and upper_ok


def normalize_solvent(solvent: str) -> str:
    """Normalize solvent names for comparison."""
    if pd.isna(solvent):
        return ""
    
    solvent = str(solvent).lower().strip()
    
    # Normalize common variants
    mapping = {
        "h2o": "water",
        "water": "water",
        "pbs": "pbs",
        "pb": "phosphate buffer",
        "phosphate buffer": "phosphate buffer",
        "tris": "tris",
        "hepes": "hepes",
        "methanol": "methanol",
        "ethanol": "ethanol",
        "chloroform": "chloroform",
        "dmso": "dmso",
        "acetone": "acetone",
        "dichloromethane": "dichloromethane",
        "toluene": "toluene",
    }
    
    return mapping.get(solvent, solvent)


def evaluate_predictions(
    df: pd.DataFrame, 
    predictions: list[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate predictions against actual test data.
    
    Returns:
        - DataFrame with predictions and evaluation results
        - Dictionary with overall metrics
    """
    results = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        result_row = {
            "peptide_code": row["PEPTIDE_CODE"],
            "morphology": row["Morphology"],
            "actual_ph": row["PH"],
            "actual_concentration_mg_ml": row["CONCENTRATION_mg ml"],
            "actual_temperature_c": row["TEMPERATURE_C"],
            "actual_solvent": row["SOLVENT"],
            "actual_time_min": row["Time (min)"],
        }
        
        # Convert concentration to log M for comparison
        result_row["actual_concentration_log_m"] = mg_ml_to_log_m(
            row["CONCENTRATION_mg ml"], 
            row["PEPTIDE_CODE"]
        )
        
        # Get prediction
        if idx >= len(predictions):
            result_row["prediction"] = ""
            result_row["parse_success"] = False
        else:
            result_row["prediction"] = predictions[idx]
            
            # Try to parse prediction
            try:
                conditions = PeptideSynthesisConditions.from_report_string(predictions[idx])
                result_row["parse_success"] = True
                
                # Store predicted values
                result_row["pred_ph_lower"] = conditions.ph.lower
                result_row["pred_ph_upper"] = conditions.ph.upper
                result_row["pred_ph_lower_incl"] = conditions.ph.lower_inclusive
                result_row["pred_ph_upper_incl"] = conditions.ph.upper_inclusive
                
                result_row["pred_conc_lower"] = conditions.concentration_log_m.lower
                result_row["pred_conc_upper"] = conditions.concentration_log_m.upper
                result_row["pred_conc_lower_incl"] = conditions.concentration_log_m.lower_inclusive
                result_row["pred_conc_upper_incl"] = conditions.concentration_log_m.upper_inclusive
                
                result_row["pred_temp_lower"] = conditions.temperature_c.lower
                result_row["pred_temp_upper"] = conditions.temperature_c.upper
                result_row["pred_temp_lower_incl"] = conditions.temperature_c.lower_inclusive
                result_row["pred_temp_upper_incl"] = conditions.temperature_c.upper_inclusive
                
                result_row["pred_solvent"] = conditions.solvent
                
                result_row["pred_time_lower"] = conditions.time_minutes.lower
                result_row["pred_time_upper"] = conditions.time_minutes.upper
                result_row["pred_time_lower_incl"] = conditions.time_minutes.lower_inclusive
                result_row["pred_time_upper_incl"] = conditions.time_minutes.upper_inclusive
                
                # Evaluate each field
                result_row["ph_correct"] = value_in_interval(
                    result_row["actual_ph"],
                    conditions.ph.lower,
                    conditions.ph.upper,
                    conditions.ph.lower_inclusive,
                    conditions.ph.upper_inclusive,
                )
                
                result_row["concentration_correct"] = value_in_interval(
                    result_row["actual_concentration_log_m"],
                    conditions.concentration_log_m.lower,
                    conditions.concentration_log_m.upper,
                    conditions.concentration_log_m.lower_inclusive,
                    conditions.concentration_log_m.upper_inclusive,
                )
                
                result_row["temperature_correct"] = value_in_interval(
                    result_row["actual_temperature_c"],
                    conditions.temperature_c.lower,
                    conditions.temperature_c.upper,
                    conditions.temperature_c.lower_inclusive,
                    conditions.temperature_c.upper_inclusive,
                )
                
                # Solvent comparison (normalized)
                actual_solvent_norm = normalize_solvent(result_row["actual_solvent"])
                pred_solvent_norm = normalize_solvent(conditions.solvent)
                result_row["solvent_correct"] = (actual_solvent_norm == pred_solvent_norm)
                
                result_row["time_correct"] = value_in_interval(
                    result_row["actual_time_min"],
                    conditions.time_minutes.lower,
                    conditions.time_minutes.upper,
                    conditions.time_minutes.lower_inclusive,
                    conditions.time_minutes.upper_inclusive,
                )
                
            except Exception as e:
                logger.warning(f"Failed to parse prediction {idx}: {e}")
                result_row["parse_success"] = False
                result_row["ph_correct"] = False
                result_row["concentration_correct"] = False
                result_row["temperature_correct"] = False
                result_row["solvent_correct"] = False
                result_row["time_correct"] = False
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    
    # Calculate metrics (only on successfully parsed predictions with non-NaN actual values)
    valid_predictions = results_df[results_df["parse_success"] == True]
    
    metrics = {
        "total_examples": len(df),
        "parse_success_rate": results_df["parse_success"].mean(),
        "num_parsed": valid_predictions.shape[0],
    }
    
    # Per-field accuracy (only count non-NaN values)
    for field in ["ph", "concentration", "temperature", "solvent", "time"]:
        field_col = f"{field}_correct"
        actual_col_map = {
            "ph": "actual_ph",
            "concentration": "actual_concentration_log_m",
            "temperature": "actual_temperature_c",
            "solvent": "actual_solvent",
            "time": "actual_time_min",
        }
        actual_col = actual_col_map[field]
        
        # Filter out rows where actual value is NaN
        valid_field = valid_predictions[~valid_predictions[actual_col].isna()]
        
        if len(valid_field) > 0:
            metrics[f"{field}_accuracy"] = valid_field[field_col].mean()
            metrics[f"{field}_count"] = len(valid_field)
        else:
            metrics[f"{field}_accuracy"] = 0.0
            metrics[f"{field}_count"] = 0
    
    # Overall accuracy (all fields correct for a prediction)
    if len(valid_predictions) > 0:
        # Only count rows where all actual values are present
        complete_rows = valid_predictions[
            ~valid_predictions["actual_ph"].isna() &
            ~valid_predictions["actual_concentration_log_m"].isna() &
            ~valid_predictions["actual_temperature_c"].isna() &
            ~valid_predictions["actual_solvent"].isna() &
            ~valid_predictions["actual_time_min"].isna()
        ]
        
        if len(complete_rows) > 0:
            all_correct = (
                complete_rows["ph_correct"] &
                complete_rows["concentration_correct"] &
                complete_rows["temperature_correct"] &
                complete_rows["solvent_correct"] &
                complete_rows["time_correct"]
            )
            metrics["overall_accuracy"] = all_correct.mean()
            metrics["complete_examples_count"] = len(complete_rows)
        else:
            metrics["overall_accuracy"] = 0.0
            metrics["complete_examples_count"] = 0
    else:
        metrics["overall_accuracy"] = 0.0
        metrics["complete_examples_count"] = 0
    
    return results_df, metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    """Pretty print evaluation metrics."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\nTotal Examples: {metrics['total_examples']}")
    print(f"Parse Success Rate: {metrics['parse_success_rate']:.2%} ({metrics['num_parsed']}/{metrics['total_examples']})")
    
    print("\n" + "-" * 60)
    print("PER-FIELD ACCURACY (on valid examples)")
    print("-" * 60)
    
    fields = ["ph", "concentration", "temperature", "solvent", "time"]
    for field in fields:
        acc_key = f"{field}_accuracy"
        count_key = f"{field}_count"
        if acc_key in metrics:
            print(f"{field.upper():<15}: {metrics[acc_key]:>6.2%} ({metrics[count_key]} examples)")
    
    print("\n" + "-" * 60)
    print("OVERALL ACCURACY (all fields correct)")
    print("-" * 60)
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%} ({metrics['complete_examples_count']} complete examples)")
    print("=" * 60 + "\n")


def main():
    """Main evaluation workflow."""
    # Setup paths
    project_root = Path(__file__).parent.parent
    test_csv = project_root / "data" / "test_split.csv"
    output_dir = project_root / "data" / "evaluation"
    output_dir.mkdir(exist_ok=True)
    
    # Load settings
    settings = Settings.from_yaml(None)
    logger.info(f"Using settings: {settings.model_dump()}")
    
    # Load test data
    df = load_test_data(test_csv)
    
    # Convert to batch requests
    requests = convert_to_batch_requests(df)
    
    # Run predictions
    logger.info("Running batch predictions...")
    predictions = predict_batch(requests, settings)
    logger.info(f"Received {len(predictions)} predictions")
    
    # Evaluate
    logger.info("Evaluating predictions...")
    results_df, metrics = evaluate_predictions(df, predictions)
    
    # Print metrics
    print_metrics(metrics)
    
    # Save results
    results_csv = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_csv, index=False)
    logger.info(f"Saved detailed results to {results_csv}")
    
    metrics_json = output_dir / "evaluation_metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_json}")
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = main()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

