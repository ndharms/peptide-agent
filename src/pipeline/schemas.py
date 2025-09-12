# src/pipeline/schemas.py
from schema import Schema, Or

# The schema package only supports validators and types, not descriptions or only_one.
# Keep constraints minimal & structural.

PEPTIDE_AGENT_RESPONSE = Schema(
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
            desciption="""
            The ideal (or target) morphology we want the peptide to self-assemble as.
            A user input.
            """,
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
            desciption="""
        An experimental condition we want to optimize such that
        it facilitates self-assembly of the peptide into the target morphology.
        The optimal pH to run the experiment at.
        """,
        ),
        "CONCENTRATION_LOG_MGML": Or(
            "(-3,-1]",
            "(-1,0]",
            "(0,1]",
            "(1,2]",
            "(2,3]",
            only_one=True,
            description="""
        An experimental condition we want to optimize such that
        it facilitates self-assembly of the peptide into the target morphology.
        The ideal concentration of reagents to use. 
        """,
        ),
        "TEMPERATURE_C": Or(
            "(0,20]",
            "(20,25]",
            "(25,37]",
            "(37,90]",
            only_one=True,
            description="""
            An experimental condition we want to optimize such that
            it facilitates self-assembly of the peptide into the target morphology.
            The ideal temperature to run the experiment at.
            """,
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
            description="""
            An experimental condition we want to optimize such that
            it facilitates self-assembly of the peptide into the target morphology.
            The ideal solvent to use in the experiment. 
            """,
        ),
        # Estimated synthesis time buckets (minutes)
        "TIME_MINUTES": Or(
            "(0,30]",
            "(30,60]",
            "(60,120]",
            "(120,240]",
            "(240, 2880]",
            only_one=True,
            description="The incubation time of the experiment. Inclusive of ",
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
PEPTIDE_AGENT_RESPONSE = PEPTIDE_AGENT_RESPONSE
