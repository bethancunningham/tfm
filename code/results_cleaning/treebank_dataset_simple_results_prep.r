library(readr)
library(dplyr)

# Model results URLs

files <- c(
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/ai-forever_mGPT-13B_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/bigscience_bloom-7b1_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/britllm_britllm-3b-v0.1_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/britllm_TransWebLLM_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/CohereLabs_tiny-aya-base_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/goldfish-models_cym_latn_5mb_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/goldfish-models_cym_latn_10mb_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/goldfish-models_cym_latn_100mb_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/goldfish-models_cym_latn_1000mb_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/LLaMAX_LLaMAX3-8B_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/meta-llama_Llama-3.1-8B_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/microsoft_phi-2_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/mistralai_Mistral-7B-v0.1_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/Qwen_Qwen3-8B_initial_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results/utter-project_EuroLLM-9B_initial_results.csv.gz"
)

# Read and concatenate in one line
df <- do.call(rbind, lapply(files, read_csv))

# Making necessary changes to df

df <- df %>%
  mutate(
    accuracy = ifelse(delta > 0, 1, 0)) %>% # Adding accuracy column (1 if incorrect NLL - correct NLL > 0 (i.e. delta)) otherwise 0)
  mutate(incorrect_form_mut_type = ifelse(
      startsWith(mutation_type, "no_"),
      sub("^no_", "", mutation_type),  # removing "no_" prefix
      "no_mutation"), # Adding incorrect_form_mut_type column
         mutation_type = ifelse(
      startsWith(mutation_type, "no_"),
      "no_mutation",
      mutation_type)
  ) %>% # Changing no_SM, no_NM, no_AM mutation_type to just no_mutation
    item = index # Adding item column (same as index in initial set)
  ) %>%
  select(
    model,
    index,
    sentence_id,
    item,
    correct_form,
    correct_sentence,
    mutation_type,
    trigger_type,
    specific_trigger,
    incorrect_form,
    incorrect_sentence,
    incorrect_form_mut_type,
    correct_nll_mean,
    incorrect_nll_mean,
    delta,
    accuracy,
    layer_nll_correct,
    layer_nll_incorrect
  ) # Reordering columns

# Writing to csv with ; as separator (write_csv2)

write_csv2(df, "results_treebank_dataset_initial.csv")
