from schema import Schema, Or, And

peptide_output_schema = Schema(
    {
        "PEPTIDE_CODE": str,
        "MORPHOLOGY": Or("None", "Fiber", "Sphere", "Aggregate"),
        "PH": Or(
            "(1,3)", "(3,5)", "(5,6.5)", "(6.5,7.5)", "(7.5,9)", "(9,11)", "(11,14)"
        ),
        "CONCENTRATION_LOG_MGML": [
            "(-3,-1)",
            "(-1,0)",
            "(0,1)",
            "(1,2)",
            "(2,3)",
        ],
        "TEMPERATURE_C": Or("37", "25"),
        "SOLVENT": Or(
            "Dimethylformamide (DMF)",
            "N-Methyl-2-pyrrolidone (NMP)",
            "Acetonitrile (ACN)",
            "Water",
            "Dichloromethane (DCM)",
            "Trifluoroacetic acid (TFA)",
            "Diethyl ether",
        ),
    }
)


paper_context = Schema(
    {
        "DOI": str,
        "PAPER_TYPE": Or("REVIEW", "RESEARCH"),
    }
)
