# src/pipeline/schemas.py
from schema import Schema, Or

# The schema package only supports validators and types, not descriptions or only_one.
# Keep constraints minimal & structural.
peptide_output_schema = Schema({
    "PEPTIDE_CODE": str,  # e.g., "FF"

    # Keep the allowed set simple. If you want a human description, store it in comments.
    "MORPHOLOGY": Or("None", "Fiber", "Sphere", "Aggregate"),

    # Buckets as strings
    "PH": Or("(1,3)", "(3,5)", "(5,6.5)", "(6.5,7.5)", "(7.5,9)", "(9,11)", "(11,14)"),

    # Buckets (your earlier file used LOG_MGML buckets)
    "CONCENTRATION_LOG_MGML": Or("(-3,-1)", "(-1,0)", "(0,1)", "(1,2)", "(2,3)"),

    # Only two temps, as strings
    "TEMPERATURE_C": Or("37", "25"),

    # Allowed solvents
    "SOLVENT": Or(
        "Dimethylformamide (DMF)",
        "N-Methyl-2-pyrrolidone (NMP)",
        "Acetonitrile (ACN)",
        "Water",
        "Dichloromethane (DCM)",
        "Trifluoroacetic acid (TFA)",
        "Diethyl ether",
    ),
    # Estimated synthesis time buckets (minutes)
    "TIME_MINUTES": Or("(<30)", "(30,60)", "(60,120)", "(120,240)", "(>240)"),
})

# Optional: validate paper metadata files
paper_context_schema = Schema({
    "DOI": str,
    "PAPER_TYPE": Or("REVIEW", "RESEARCH"),
})

# Backward compat alias if other modules still import this name:
PEPTIDE_AGENT_RESPONSE = peptide_output_schema
