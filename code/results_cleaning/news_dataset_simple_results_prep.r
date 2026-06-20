library(readr)
library(dplyr)
library(tidyverse)

# Results file URLs

files <- c(
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/goldfish-models_cym_latn_5mb_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/goldfish-models_cym_latn_10mb_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/goldfish-models_cym_latn_100mb_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/goldfish-models_cym_latn_1000mb_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/britllm-3b-v0.1_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/bloom-7b1_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/Llama-3.1-8B_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/mGPT-13B_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/Mistral-7B-v0.1_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/phi-2_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/EuroLLM-9B_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/Qwen3-8B_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/LLaMAX3-8B_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/tiny-aya-base_smalldataset_initial.csv",
  "https://raw.githubusercontent.com/bethancunningham/tfm/main/initial_results_small/TransWebLLM_smalldataset_initial.csv"
)

# Reading and concatenating files to create df of all results

df <- do.call(rbind, lapply(files, read_csv))

# Making necessary changes to df

df <- df %>%
  mutate(
    accuracy = ifelse(delta > 0, 1, 0), # Adding accuracy column (1 if incorrect NLL - correct NLL > 0 (i.e. delta), otherwise 0)
    
    incorrect_form_mut_type = ifelse(
      startsWith(mutation_type, "no_"),
      sub("^no_", "", mutation_type),
      "no_mutation"
    ), # Creating incorrect_form_mut_type column (e.g. no_SM in mutation_type becomes SM)
    
    mutation_type = ifelse(
      startsWith(mutation_type, "no_"),
      "no_mutation",
      mutation_type
    ) # Changing no_SM, no_NM, no_AM mutation_type to just no_mutation
    
  ) %>%
  select(-item) %>% # Removing corrupt item column
  mutate(item = index) %>% # Copying index column and using as item (same in this case, i.e. simple set not expanded)
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

write_csv2(df, "results_news_dataset_initial.csv")