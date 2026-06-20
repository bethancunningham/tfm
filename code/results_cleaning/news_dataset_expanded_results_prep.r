library(readr)
library(dplyr)

# Results file URLs

files <- c(
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/goldfish-models_cym_latn_5mb_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/goldfish-models_cym_latn_10mb_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/goldfish-models_cym_latn_100mb_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/goldfish-models_cym_latn_1000mb_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/britllm-3b-v0.1_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/bloom-7b1_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/Llama-3.1-8B_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/mGPT-13B_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/Mistral-7B-v0.1_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/phi-2_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/EuroLLM-9B_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/Qwen3-8B_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/LLaMAX3-8B_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/tiny-aya-base_smalldataset_expanded.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/expanded_results_small/TransWebLLM_smalldataset_expanded.csv"
)

# Reading and concatenating files to create df of all results

df <- do.call(rbind, lapply(files, read_csv))

# Making necessary changes to df

df <- df %>%
  mutate(
    accuracy = ifelse(delta > 0, 1, 0), # Adding accuracy column (1 if incorrect NLL - correct NLL > 0 (i.e. delta), otherwise 0)
    
    incorrect_form_mut_type = ifelse(
      incorrect_form_mut_type == "none",
      "no_mutation",
      incorrect_form_mut_type
    ), # Changing "none" in this column to "no_mutation" to match others
    
    mutation_type = ifelse(
      startsWith(mutation_type, "no_"),
      "no_mutation",
      mutation_type
    ) # Changing no_SM, no_NM, no_AM mutation_type to just no_mutation
    )

# Getting original dataset and pulling correct item column

original <- read_csv("https://raw.githubusercontent.com/bethancunningham/tfm/main/datasets/expanded_news_dataset.csv")

original_subset <- original %>%
  rename(index = "...1") %>%
  select(index, item)

# Dropping corrupted item column and replacing with the correct one

df <- df %>%
  select(-item) %>%
  left_join(original_subset, by = "index") %>% # Adding item number according to index
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

write_csv2(df, "results_news_dataset_expanded.csv")