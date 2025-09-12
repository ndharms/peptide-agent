# src/pipeline/schemas.py
from schema import Schema, Or

# The schema package only supports validators and types, not descriptions or only_one.
# Keep constraints minimal & structural.

# PEPTIDE_AGENT_RESPONSE = Schema(
peptide_output_schema = Schema(
    {
        "PEPTIDE_CODE": Schema(
            str,
            description="""
            A multi-letter code which defines the peptide we want to self-assemble. i.e., FF
            A user input.
            """,
        ),
        "MORPHOLOGY": Or(
            "none",
            "fiber",
            "sphere",
            "aggregate",
            only_one=True,
        ),
        "PH": Or(
            "(1,3]",
            "(3,5]",
            "(5,6.5]",
            "(6.5,7.5]",
            "(7.5,9]",
            "(9,11]",
            "(11,14]",
            only_one=True,
        ),
        "CONCENTRATION_LOG_MGML": Or(
            "(-3,-1]",
            "(-1,0]",
            "(0,1]",
            "(1,2]",
            "(2,3]",
            only_one=True,
        ),
        "TEMPERATURE_C": Or(
            "(0,20]",
            "(20,25]",
            "(25,37]",
            "(37,90]",
            only_one=True,
        ),
        "SOLVENT": Or(
            "Dimethylformamide (DMF)",
            "N-Methyl-2-pyrrolidone (NMP)",
            "Acetonitrile (ACN)",
            "Water",
            "Dichloromethane (DCM)",
            "Trifluoroacetic acid (TFA)",
            "Diethyl ether",
            only_one=True,
        ),
        # Estimated synthesis time buckets (minutes)
        "TIME_MINUTES": Or(
            "(0,30]",
            "(30,60]",
            "(60,120]",
            "(120,240]",
            "(240, 2880]",
            only_one=True,
        ),
    },
    description="""
    This is the output schema-definition for the LLM response 
    when providing a guess of the ideal experimental conditions.
    """,
)

# Optional: validate paper metadata files
paper_context_schema = Schema(
    {
        "DOI": str,
        "PAPER_TYPE": Or("REVIEW", "RESEARCH"),
    }
)

# Backward compat alias if other modules still import this name:
# PEPTIDE_AGENT_RESPONSE = PEPTIDE_AGENT_RESPONSE
