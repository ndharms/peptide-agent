# MIT's `peptide-agent`

This project is designed to enable a user to predict the most probable method for success when synthesizing peptides.


# Background Information

## The initial use case

Kubra, one of our teammates, synthesized a peptide as part of her PhD dissertation.
During her defense, she was asked, "why did you choose that particular synthesis route for this peptide?"
In actuality, there are many different experimental procedures for creating many different morphologies of this particular peptide.
Her answer was "well, there were many methods... I chose one and it worked."

Although this answer satisfies "why" the decision was made, 
our user needs a more robust way to choose a synthesis route with the greatest probability of success.


## The proposed solution

We propose an LLM-based solution to address our user's need.
This solution will take a user's requested peptide and a target structural assembly (i.e., peptide morphology) as input.
First, the LLM will search the internet for relevant papers which contain synthesis procedures for identical and/or similar peptides.
This will serve as a base understanding which mirrors a "literature review" which a scientist would perform.
Secondly, we will manually curate examples of successful synthesis and store these in the `data` directory.
The LLM will reference these examples within the context of the task to predict multiple guesses 
for the experimental conditions required for the synthesis.
These conditions include (but are not limited to) pH, Concentration, Temperature, Solvent, and Time.

With these predicted conditions and the "literature review," the LLM will construct a "Peptide Synthesis Profile" for the peptide of interest. 


# Solution Design

## Architecture and Repo Design

We'll be designing the repo with the following format:

- `./src`: a directory for the code
- `./src/pipeline`: a subdirectory for the pipeline specific code
- `./src/pipeline/prompts`: a sub-subdirectory for any relevant prompts
- `./scripts`: a directory for *ad hoc* validation scripts
- `./data`: a directory for our multi-shot examples and any other relevant data.
- `setup.py`: the file which defines the required packages for this project, references `./src/pipeline/__version__.py`


## Expected outputs

### Pt 1) Literature searches

For this, we are going to ask the LLM to perform a literature review for a particular peptide.
The LLM will use a well-defined prompt describing how to perform a literature search for peptides.
It will focus on 

### Pt 2) Experimental Condition guesses

### Pt 3) Combining results


## Why this will work

- **Language-focused problem**:
- **Multi-shot learning**: This solution will rely on [multi-shot learning](https://medium.com/@anmoltalwar/multi-shot-prompting-15a7c4b8b78e)
with manually curated examples. 



