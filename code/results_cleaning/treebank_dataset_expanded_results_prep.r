library(readr)
library(dplyr)

# Results file URLs

files <- c(
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/goldfish_5mb_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/goldfish_10mb_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/goldfish_100mb_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/goldfish_1000mb_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/britllm_3B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/bloom_7B1_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/llama_3.1_8B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/mgpt_13B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/mistral_7B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/phi_2_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/EuroLLM-9B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/Qwen3-8B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/LLaMAX3-8B_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/CohereLabs_tiny-aya-base_expanded_results.csv.gz",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results/britllm_TransWebLLM_expanded_results.csv.gz"
)

# Reading and concatenating files to create df of all results

df <- do.call(rbind, lapply(files, read_csv))

# Getting original dataset with index column

original <- read_csv("https://raw.githubusercontent.com/bethancunningham/tfm/main/datasets/final_df_expanded_with_index.csv")

# Adding accuracy column (1 if incorrect NLL - correct NLL > 0 (i.e. delta), otherwise 0)

df <- df %>%
  mutate(accuracy = ifelse(delta > 0, 1, 0))

# Taking desired columns from original dataset

original_subset <- original %>% 
  select(index, token_id, incorrect_form_mut_type, correct_form, incorrect_form)

# Adding columns to df, matching values according to index

df <- left_join(df, original_subset, by = "index")

# Creating item column

df <- df %>%
  arrange(index) %>%
  mutate(item = consecutive_id(sentence_id, token_id) - 1) %>%
  arrange(model)

# Making necessary changes to df

df <- df %>%
  mutate(mutation_type = ifelse(startsWith(mutation_type, "no_"), "no_mutation", mutation_type),
    incorrect_form_mut_type = ifelse(
    incorrect_form_mut_type == "none",
    "no_mutation",
    incorrect_form_mut_type)) %>% # Changing "none" in this column to "no_mutation" to match others)
  rename(correct_sentence = sentence) %>% # Renaming sentence column
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

write_csv2(df, "results_treebank_dataset_expanded.csv")