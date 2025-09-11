from schema import Schema, Or, And

peptide_output_schema = Schema(
    {
        "PEPTIDE_CODE": str,
        "MORPHOLOGY": Or("None", "Fiber", "Sphere", "Aggregate"),
        "PH": int,
        "CONCENTRATION_LOG_M": int,
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
