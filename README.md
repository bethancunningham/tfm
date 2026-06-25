# Language Models and Initial Consonant Mutation in Welsh: A Minimal-Pair Benchmark and Interpretability Analysis

This repository contains the code and data for a master's dissertation investigating how large language models handle **Initial Consonant Mutation (ICM)** in Welsh — a grammatical phenomenon where the first consonant of a word changes depending on its lexical or morphosyntactic context.

## 🔍 Contents
 
| Component | Description |
|---|---|
| 🏗️ **Dataset pipeline** | Pair generation from the Welsh UD Treebank and Welsh news corpus |
| 📊 **Behavioural benchmark** | ~7,000 minimal pairs scored by negative log likelihood (NLL) across 15 models |
| 🔬 **Logit lens analysis** | Layer-by-layer NLL tracking to shed light on ICM processing |
| 📈 **Statistical models** | GLMMs and LMMs to examine effects of model, trigger type, mutation type and layer |
 
---
 
## 🤖 Models evaluated

`goldfish-models/cym_latn_5mb`, `goldfish-models/cym_latn_10mb`, `goldfish-models/cym_latn_100mb`, `goldfish-models/cym_latn_1000mb`, `britllm/britllm-3b-v0.1`, 
`meta-llama/Llama-3.1-8B`, `ai-forever/mGPT-13B`, `microsoft/phi-2`, `mistralai/Mistral-7B-v0.1`, `bigscience/bloom-7b1`, `britllm/TransWebLLM`, `CohereLabs/tiny-aya-base`, 
`utter-project/EuroLLM-9B`, `Qwen/Qwen3-8B`, `LLaMAX/LLaMAX3-8B`

## 🗂️ Workflow
### Phase 1 · Dataset creation
 
- [Treebank dataset creation script](code/dataset_creation/dataset_creation_treebank.py) — takes [Treebank source files and list of feminine singular nouns](source_files), outputs files for manual annotation where automatic annotation failed, takes the [completed manual annotations](manual_annotations), then outputs the final [Simple](datasets/initial_treebank_dataset.csv) and [Expanded](datasets/expanded_treebank_dataset.csv) Treebank datasets
- [News dataset creation script](code/dataset_creation/dataset_creation_news.py) — takes the [News sentences file](datasets/sentences_news_articles_semicolon.csv) and outputs the final [Simple](datasets/initial_news_dataset.csv) and [Expanded](datasets/expanded_news_dataset.csv) News datasets
### Phase 2 · Pipeline to assign NLL
 
- [Treebank pipeline](code/pipeline/Pipeline_tfm_treebank_dataset.ipynb) — loads all models, assigns NLL to sentences in minimal pairs, and runs the logit lens. Takes the [Expanded](datasets/expanded_treebank_dataset.csv) and [Simple](datasets/initial_treebank_dataset.csv) Treebank datasets and outputs [expanded results](expanded_results), [simple results](initial_results), and [logit lens results](logit_lens_results)
- [News pipeline](code/pipeline/Pipeline_tfm_news_dataset.ipynb) — loads all models and assigns NLL to sentences in minimal pairs. Takes the [Expanded](datasets/expanded_news_dataset.csv) and [Simple](datasets/initial_news_dataset.csv) News datasets and outputs [expanded results](expanded_results_small) and [simple results](initial_results_small)
### Phase 3 · Data cleaning
 
- [Results cleaning scripts](code/results_cleaning) — prepare results for analysis. Takes [Simple](initial_results) and [Expanded](expanded_results) Treebank results and [Simple](initial_results_small) and [Expanded](expanded_results_small) News results, and outputs [clean results](clean_results)
### Phase 4 · Data analysis
 
- [Data analysis script](code/data_analysis) — analyses data, fits statistical models, and produces plots and tables
---
 
## 📁 Repository structure
 
```
├── code/
│   ├── dataset_creation/       # Pair generation scripts
│   ├── pipeline/               # NLL scoring notebooks (Treebank + News)
│   ├── results_cleaning/       # Cleaning scripts
│   └── data_analysis/          # Statistical modelling, plots, and tables
├── datasets/                   # Input datasets (Treebank + News)
├── source_files/               # Raw treebank files, noun lists
├── manual_annotations/         # Human-annotated edge cases
├── initial_results/            # Raw Simple results (Treebank)
├── expanded_results/           # Raw Expanded results (Treebank)
├── initial_results_small/      # Raw Simple results (News)
├── expanded_results_small/     # Raw Expanded results (News)
├── logit_lens_results/         # Per-layer NLL outputs
├── clean_results/              # Clean, merged results
├── error_analysis/             # Error analysis outputs
└── human_test/                 # Human evaluation data
```
 
---
 
## 📄 Citation
 
If you use this benchmark or code, please cite:
 
```bibtex
@mastersthesis{cunningham2026welsh,
  title  = {Language Models and Initial Consonant Mutation in Welsh: A Minimal-Pair Benchmark and Interpretability Analysis},
  author = {Cunningham, Bethan},
  year   = {2026},
  school = {[Universitat Pompeu Fabra]}
}
