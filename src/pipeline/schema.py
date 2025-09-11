from schema import Schema, Or, And

peptide_output_schema = Schema(
    {
        "PEPTIDE_CODE": str,
        "TARGET_STRUCTURAL_ASSEMBLY": str,
        "PH": And(int, lambda x: x > 0),
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

example_context = Schema(
    {peptide_output_schema.schema | {"OUTCOME": Or("SUCCESS", "FAILURE")}}
)


paper_context = Schema(
    {
        "DOI": str,
        "PAPER_TYPE": Or("REVIEW", "RESEARCH"),
    }
)
