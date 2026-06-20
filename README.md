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
- [Treebank dataset creation script](code/dataset_creation/dataset_creation_treebank.py): takes [Treebank source files and list of feminine singular nouns](source_files), outputs the files for manual annotation where automatic annotation failed, takes the [completed manual annotations](manual_annotations), then outputs the final [Simple Treebank dataset](datasets/initial_treebank_dataset.py) and [Expanded Treebank dataset](datasets/expanded_treebank_dataset.py)
- [News dataset creation script](code/dataset_creation/dataset_creation_news.py): takes [News sentences file](datasets/sentences_news_articles_semicolon.csv), outputs the final [Simple News dataset](datasets/initial_news_dataset.csv) and [Expanded News dataset](datasets/expanded_news_dataset.csv)

### Phase 2 - Pipeline to assign NLL
- [Treebank pipeline](code/pipeline/Pipeline_tfm_treebank_dataset.ipynb) to load all models and assign NLL to sentences in minimal pairs + logit lens implementation. Takes [Expanded Treebank dataset](datasets/expanded_treebank_dataset.csv) and [Simple Treebank dataset](datasets/initial_treebank_dataset.csv) and outputs [Expanded results](expanded_results), [Simple results](initial_results) and [logit lens results](logit_lens_results)
- [News pipeline](code/pipeline/Pipeline_tfm_news_dataset.ipynb) to load all models and assign NLL to sentences in minimal pairs. Takes [Expanded News dataset](datasets/expanded_news_dataset.csv) and [Simple Treebank dataset](datasets/initial_treebank_dataset.csv) and outputs [Expanded results](expanded_results_small), [Simple results](initial_results_small)

### Phase 3 - Data cleaning
- [Results cleaning scripts](code/results_cleaning) to prepare results for analysis. Takes [Simple Treebank results](initial_results), [Expanded Treebank results](expanded_results), [Simple News results](initial_results_small) and [Expanded News results](expanded_results_small) and outputs [clean results](clean_results)

### Phase 4 - Data analysis
- 
