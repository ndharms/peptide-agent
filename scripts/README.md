# Evaluation Scripts

This directory contains scripts for evaluating the peptide agent's performance.

## eval_test_data.py

Evaluation script that tests the peptide agent on held-out test data and computes classification metrics.

### What it does

1. **Loads test data** from `data/test_split.csv`
2. **Converts to batch format** for `predict_batch()` in `main.py`
3. **Runs predictions** using the configured LLM model
4. **Parses predictions** into structured conditions
5. **Evaluates accuracy** by comparing:
   - Whether actual pH falls within predicted pH interval
   - Whether actual concentration (converted to log M) falls within predicted interval
   - Whether actual temperature falls within predicted interval
   - Whether actual solvent matches predicted solvent (normalized comparison)
   - Whether actual time falls within predicted time interval
6. **Computes metrics**:
   - Parse success rate
   - Per-field accuracy (pH, concentration, temperature, solvent, time)
   - Overall accuracy (all fields correct)

### Usage

```bash
# Make sure you're in the project root and have activated the virtual environment
cd /Users/nathanharms/PycharmProjects/peptide-agent
source venv/bin/activate

# Set required environment variables
export GEMINI_API_KEY="your-api-key-here"

# Run evaluation
python scripts/eval_test_data.py
```

### Output

The script creates two files in `data/evaluation/`:

1. **evaluation_results.csv** - Detailed per-example results with:
   - Input data (peptide code, morphology, actual conditions)
   - Predictions (parsed intervals and values)
   - Correctness flags for each field

2. **evaluation_metrics.json** - Summary metrics including:
   - Parse success rate
   - Per-field accuracy percentages
   - Overall accuracy
   - Counts of valid examples per field

### Configuration

The script uses the default `Settings` configuration. You can override settings via environment variables:

```bash
export PEPTIDE_LLM_MODEL="gemini-2.5-pro"
export PEPTIDE_TOP_K=10
export PEPTIDE_BATCH_SIZE=40
```

### Metrics Explained

- **Parse Success Rate**: Percentage of predictions that could be successfully parsed into the expected format
- **Per-field Accuracy**: For each field (pH, concentration, temperature, solvent, time), the percentage of examples where:
  - The actual value falls within the predicted interval (for numeric fields)
  - The actual solvent matches the predicted solvent (for solvent field)
- **Overall Accuracy**: Percentage of examples where ALL fields are correct simultaneously

### Notes

- Only examples with non-NaN actual values are included in accuracy calculations
- Concentration is converted from mg/ml (in the CSV) to log M for comparison with predictions
- Molecular weight is estimated as ~110 Da per amino acid for conversion
- Solvent names are normalized (e.g., "H2O" → "water", "PB" → "phosphate buffer")

