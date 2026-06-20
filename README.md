# Language Models and Initial Consonant Mutation in Welsh: A Minimal-Pair Benchmark and Interpretability Analysis

This repository contains the code and data for a master's dissertation investigating how large language models handle **Initial Consonant Mutation (ICM)** in Welsh — a grammatical phenomenon where the first consonant of a word changes depending on its lexical or morphosyntactic context.

The project combines:

- A **behavioural evaluation**  using negative log likelihood (NLL) scoring on ~7,000 minimal pairs derived from the Welsh Treebank.
- A **mechanistic interpretability analysis** using the logit lens technique, tracking how NLL evolves across model layers.

## Models evaluated

`goldfish-models/cym_latn_5mb`, `goldfish-models/cym_latn_10mb`, `goldfish-models/cym_latn_100mb`, `goldfish-models/cym_latn_1000mb`, `britllm/britllm-3b-v0.1`, 
`meta-llama/Llama-3.1-8B`, `ai-forever/mGPT-13B`, `microsoft/phi-2`, `mistralai/Mistral-7B-v0.1`, `bigscience/bloom-7b1`, `britllm/TransWebLLM`, `CohereLabs/tiny-aya-base`, 
`utter-project/EuroLLM-9B`, `Qwen/Qwen3-8B`, `LLaMAX/LLaMAX3-8B`

## Workflow
### Phase 1 - Dataset creation
- [Treebank dataset creation script](code/dataset_creation/dataset_creation_treebank.py): takes [Treebank source files and list of feminine singular nouns](source_files), outputs the files for manual annotation where automatic annotation failed, takes the [completed manual annotations](manual_annotations), then outputs the final [Treebank Simple dataset](initial_treebank_dataset.py) and [Treebank Expanded dataset](expanded_treebank_dataset.py)
