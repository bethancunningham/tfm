library(dplyr)
library(flextable)
library(ggplot2)
library(tidyr)
library(magrittr)
library(data.table)
library(scales)
library(lme4)
library(marginaleffects)
library(tibble)
library(summarytools)
library(officer)
library(magick)
library(tidyverse)
library(reformulas)

# --- 1. Data frames ---

# Model metadata to add to dfs

model_metadata <- tibble(
  model = c(
    "goldfish-models/cym_latn_1000mb",
    "goldfish-models/cym_latn_100mb",
    "britllm/britllm-3b-v0.1",
    "goldfish-models/cym_latn_10mb",
    "LLaMAX/LLaMAX3-8B",
    "goldfish-models/cym_latn_5mb",
    "britllm/TransWebLLM",
    "meta-llama/Llama-3.1-8B",
    "CohereLabs/tiny-aya-base",
    "Qwen/Qwen3-8B",
    "mistralai/Mistral-7B-v0.1",
    "utter-project/EuroLLM-9B",
    "ai-forever/mGPT-13B",
    "bigscience/bloom-7b1",
    "microsoft/phi-2"
  ),
  training_data = c(
    "monolingual", "monolingual", "multilingual_welsh", "monolingual",
    "multilingual_welsh", "monolingual", "multilingual_welsh",
    "multilingual_no_celtic", "multilingual_welsh", "multilingual_no_celtic",
    "english", "multilingual_irish", "multilingual_no_celtic",
    "multilingual_no_celtic", "english"
  ),
  model_size = c(
    "125M", "125M", "3B", "39M", "8B", "39M", "1.3B",
    "8B", "3.3B", "8B", "7B", "9B", "13B", "7B", "2.7B"
  ),
  layers = c(12, 12, 32, 4, 32, 4, 24, 32, 36, 36, 32, 42, 40, 30, 32)
)

# Model labels to change

model_labels <- c(
  "goldfish-models/cym_latn_1000mb" = "Goldfish 1000mb",
  "goldfish-models/cym_latn_100mb"  = "Goldfish 100mb",
  "goldfish-models/cym_latn_10mb"   = "Goldfish 10mb",
  "goldfish-models/cym_latn_5mb"    = "Goldfish 5mb",
  "britllm/britllm-3b-v0.1"         = "BritLLM 3B",
  "britllm/TransWebLLM"             = "TransWebLLM",
  "LLaMAX/LLaMAX3-8B"              = "LLaMAX3 8B",
  "meta-llama/Llama-3.1-8B"        = "Llama 3.1 8B",
  "CohereLabs/tiny-aya-base"        = "Tiny Aya",
  "Qwen/Qwen3-8B"                   = "Qwen3 8B",
  "mistralai/Mistral-7B-v0.1"       = "Mistral 7B",
  "ai-forever/mGPT-13B"             = "mGPT 13B",
  "utter-project/EuroLLM-9B"        = "EuroLLM-9B",
  "bigscience/bloom-7b1"            = "Bloom 7B1",
  "microsoft/phi-2"                 = "Phi 2"
)

# Reading in News dataset (small, leaking impossible) in simple (initial) and expanded versions

df_news_initial <- read.csv2("https://raw.githubusercontent.com/bethancunningham/tfm/main/clean_results/results_news_dataset_initial.csv")
options(timeout = 300)

df_news_expanded <- read.csv2("https://raw.githubusercontent.com/bethancunningham/tfm/main/clean_results/results_news_dataset_expanded.csv")
options(timeout = 300)

# Reading in Treebank dataset (large, leaking possible) in simple (initial) and expanded versions. Need to download because large and compressed

temp1 <- tempfile(fileext = ".csv.gz")
options(timeout = 300)
download.file("https://raw.githubusercontent.com/bethancunningham/tfm/master/results_treebank_dataset_initial.csv.gz", temp1, mode = "wb")
df_treebank_initial <- read.csv2(temp1)

temp2 <- tempfile(fileext = ".csv.gz")
options(timeout = 300)
download.file("https://raw.githubusercontent.com/bethancunningham/tfm/master/results_treebank_dataset_expanded.csv.gz", temp2, mode = "wb")
df_treebank_expanded <- read.csv2(temp2)

# Creating function to clean df (making factors factors, numbers numeric, etc. Changing model labels, adding model metadata)

clean_df <- function(df) {
  df |>
    left_join(model_metadata, by = "model") |>
    mutate(
      model                = recode(model, !!!model_labels) |> as.factor(),
      index                = as.factor(index),
      sentence_id          = as.factor(sentence_id),
      item                 = as.factor(item),
      correct_form         = as.character(correct_form),
      correct_sentence     = as.character(correct_sentence),
      mutation_type        = as.factor(mutation_type),
      trigger_type         = as.factor(trigger_type),
      specific_trigger     = as.factor(specific_trigger),
      incorrect_form       = as.character(incorrect_form),
      incorrect_sentence   = as.character(incorrect_sentence),
      incorrect_form_mut_type = as.factor(incorrect_form_mut_type),
      training_data        = as.factor(training_data),
      model_size           = as.factor(model_size),
      layers               = as.integer(layers),
      accuracy             = as.numeric(accuracy),
      correct_nll_mean     = round(as.numeric(correct_nll_mean),   2),
      incorrect_nll_mean   = round(as.numeric(incorrect_nll_mean), 2),
      delta                = round(as.numeric(delta),              2),
      nll_distance         = round(as.numeric(nll_distance),       2)
    )
}

# Applying functions to dfs

df_news_initial      <- clean_df(df_news_initial)
df_news_expanded     <- clean_df(df_news_expanded)
df_treebank_initial  <- clean_df(df_treebank_initial)
df_treebank_expanded <- clean_df(df_treebank_expanded)

# --- 2. Accuracy tables ---

# Creating accuracy table for news_initial dataset (one pair per item, mutation vs non-mutation. Small dataset, no leaking)

tbl_news_initial <- df_news_initial |>
  group_by(model) |>
  summarise(
    accuracy_overall  = mean(accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Creating accuracy table for news_expanded dataset (pair for every incorrect option. Small dataset, no leaking)

tbl_news_expanded <- df_news_expanded |>
  group_by(model) |>
  summarise(
    accuracy_overall  = mean(accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Creating accuracy table for treebank_initial dataset (one pair per item, mutation vs non-mutation. Large dataset, possible leaking)

tbl_treebank_initial <- df_treebank_initial |>
  group_by(model) |>
  summarise(
    accuracy_overall  = mean(accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Creating accuracy table for treebank_expanded dataset (pair for every incorrect option. Large dataset, possible leaking)

tbl_treebank_expanded <- df_treebank_expanded |>
  group_by(model) |>
  summarise(
    accuracy_overall  = mean(accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Creating accuracy table for news_expanded dataset based on accuracy per item (1 if all pairs correct, 0 otherwise)

tbl_news_accuracy_per_item <- df_news_expanded |>
  group_by(model, item) |>
  summarise(
    item_accuracy = as.integer(all(accuracy == 1)),
    mutation_type = mutation_type,
    trigger_type  = trigger_type,
    .groups = "drop"
  ) |>
  group_by(model) |>
  summarise(
    accuracy_overall = mean(item_accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(item_accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(item_accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(item_accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(item_accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(item_accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(item_accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(item_accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Creating accuracy table for treebank_expanded dataset based on accuracy per item (1 if all pairs correct, 0 otherwise)

tbl_treebank_accuracy_per_item <- df_treebank_expanded |>
  group_by(model, item) |>
  summarise(
    item_accuracy = as.integer(all(accuracy == 1)),
    mutation_type = mutation_type,
    trigger_type  = trigger_type,
    .groups = "drop"
  ) |>
  group_by(model) |>
  summarise(
    accuracy_overall = mean(item_accuracy),
    # Accuracy by mutation type
    accuracy_SM       = mean(item_accuracy[mutation_type == "SM"]),
    accuracy_NM       = mean(item_accuracy[mutation_type == "NM"]),
    accuracy_AM       = mean(item_accuracy[mutation_type == "AM"]),
    accuracy_no_mut = mean(item_accuracy[mutation_type == "no_mutation"]),
    # Accuracy by trigger type
    accuracy_L        = mean(item_accuracy[trigger_type == "L"]),
    accuracy_MS       = mean(item_accuracy[trigger_type == "MS"]),
    accuracy_no_trig  = mean(item_accuracy[trigger_type == "no_mutation"]),
    .groups = "drop"
  ) |>
  arrange(desc(accuracy_overall))

# Descriptive stats for Treebank per-item accuracy table

make_descr_flextable <- function(tbl, path) {
  
  tbl_selected <- tbl[, c("accuracy_overall", "accuracy_SM", "accuracy_NM", "accuracy_AM",
                          "accuracy_no_mut", "accuracy_L", "accuracy_MS", "accuracy_no_trig")]
  
  descr_raw <- descr(tbl_selected, headings = FALSE, stats = c("mean", "sd", "min", "med", "max"))
  
  descr_tbl <- as.data.frame(descr_raw)
  descr_tbl$stat <- rownames(descr_tbl)
  rownames(descr_tbl) <- NULL
  
  descr_tbl$stat <- dplyr::recode(descr_tbl$stat,
                                  "Mean"    = "Mean",
                                  "Std.Dev" = "SD",
                                  "Min"     = "Min",
                                  "Median"  = "Median",
                                  "Max"     = "Max"
  )
  
  descr_tbl <- descr_tbl[, c("stat", "accuracy_overall", "accuracy_SM", "accuracy_NM",
                             "accuracy_AM", "accuracy_no_mut", "accuracy_L",
                             "accuracy_MS", "accuracy_no_trig")]
  
  better_tbl_header <- data.frame(
    col_keys = c("stat", "accuracy_overall", "accuracy_SM", "accuracy_NM",
                 "accuracy_AM", "accuracy_no_mut", "accuracy_L",
                 "accuracy_MS", "accuracy_no_trig"),
    line2 = c("", "Overall accuracy", rep("Acc. by mutation type", 4), rep("Acc. by trigger type", 3)),
    line3 = c("", "Overall accuracy", "SM", "NM", "AM", "None", "L", "MS", "None")
  )
  
  ft <- flextable(descr_tbl, col_keys = better_tbl_header$col_keys) |>
    set_header_df(mapping = better_tbl_header, key = "col_keys") |>
    merge_h(part = "header") |>
    merge_v(part = "header") |>
    theme_booktabs() |>
    align(align = "center", part = "all") |>
    align(j = "stat", align = "left", part = "all") |>
    colformat_double(j = -1, digits = 2) |>
    autofit()
  
  save_as_docx(ft, path = path)
  
  ft
}

make_descr_flextable(tbl_treebank_accuracy_per_item, "descriptive_stats.docx")

# Creating df with one row per item per model and item accuracy column (treebank expanded) for plots

df_treebank_item <- df_treebank_expanded |>
  group_by(model, item) |>
  summarise(
    item_accuracy   = as.integer(all(accuracy == 1)),
    mutation_type   = first(mutation_type),
    trigger_type    = first(trigger_type),
    sentence_id     = first(sentence_id),
    training_data   = first(training_data),
    model_size      = first(model_size),
    layers          = first(layers),
    .groups = "drop"
  )

# Joining metadata to table

tbl_treebank_accuracy_per_item <- tbl_treebank_accuracy_per_item |>
  left_join(
    df_treebank_item |> select(model, training_data, layers, model_size) |> distinct(),
    by = "model"
  ) |>
  relocate(training_data, layers, model_size, .after = model)

# Creating function to make flextables with gradient colours (green good, yellow middling, red bad)

make_accuracy_flextable_colours <- function(tbl, path) {
  
  better_tbl <- tbl |>
    select(model, starts_with("accuracy")) |>
    mutate(across(where(is.factor), as.character)) |>
    mutate(across(-model, as.numeric))
  
  better_tbl_header <- data.frame(
    col_keys = c("model", "accuracy_overall", "accuracy_SM", "accuracy_NM", 
                 "accuracy_AM", "accuracy_no_mut", "accuracy_L", 
                 "accuracy_MS", "accuracy_no_trig"),
    line2 = c("Model", "Overall accuracy", rep("Acc. by mutation type", 4), rep("Acc. by trigger type", 3)),
    line3 = c("Model", "Overall accuracy", "SM", "NM", "AM", "None", "L", "MS", "None")
  )
  
  colourer <- function(x) {
    col_numeric(
      palette  = c("#e84a57", "#e8e34a", "#56d667"),
      domain   = c(0, 1),
      na.color = "transparent"
    )(x)
  }
  
  ft <- flextable(better_tbl, col_keys = better_tbl_header$col_keys) |>
    set_header_df(mapping = better_tbl_header, key = "col_keys") |>
    merge_v(part = "header", j = 1) |>
    merge_v(part = "header", j = 2) |>
    merge_h(part = "header", i = 1) |>
    theme_booktabs(bold_header = TRUE) |>
    align(align = "center", part = "all") |>
    bg(bg = colourer, j = ~ . - model, part = "body") |>
    vline(j = c(1, 2, 6), border = fp_border_default()) |>
    colformat_double(
      j      = c("accuracy_overall", "accuracy_SM", "accuracy_NM", "accuracy_AM",
                 "accuracy_no_mut", "accuracy_L", "accuracy_MS", "accuracy_no_trig"),
      digits = 2
    ) |>
    width(j = 1,   width = 1.4) |>
    width(j = 2,   width = 0.8) |>
    width(j = 3:6, width = 0.5) |>
    width(j = 7:9, width = 0.5) |>
    font(fontname = "Times New Roman", part = "all") |>
    border_outer(border = fp_border_default())
  
  # Saving table as image
  tmp_table <- tempfile(fileext = ".png")
  save_as_image(ft, path = tmp_table, webshot = "webshot2")
  
  # Legend
  gradient_df <- data.frame(x = seq(0, 1, length.out = 500), y = 1)
  legend_plot <- ggplot(gradient_df, aes(x = x, y = y, fill = x)) +
    geom_tile() +
    scale_fill_gradientn(colours = c("#e84a57", "#e8e34a", "#56d667")) +
    scale_x_continuous(
      breaks = seq(0, 1, by = 0.1),
      labels = paste0(seq(0, 1, by = 0.1)),
      expand = c(0, 0)
    ) +
    scale_y_continuous(expand = c(0, 0)) +
    labs(x = NULL, y = NULL) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.y     = element_blank(),
      axis.ticks.y    = element_blank(),
      panel.grid      = element_blank(),
      axis.text.x     = element_text(family = "Times New Roman", size = 8)
    )
  
  tmp_legend <- tempfile(fileext = ".png")
  ggsave(tmp_legend, legend_plot, width = 5, height = 0.6, dpi = 300)
  
  # Stitching together
  table_img  <- magick::image_read(tmp_table)
  legend_img <- magick::image_read(tmp_legend)
  legend_img <- magick::image_resize(legend_img, paste0(magick::image_info(table_img)$width, "x"))
  
  magick::image_append(c(table_img, legend_img), stack = TRUE) |>
    magick::image_write(path)
  
  ft
}

# Apply colourful flextable function to per_item tables

make_accuracy_flextable_colours(tbl_news_accuracy_per_item, "flextable_colours_accuracy_item_news.png")
make_accuracy_flextable_colours(tbl_treebank_accuracy_per_item, "flextable_colours_accuracy_item_treebank.png")

# Function to make same flextable but without colours for all dataset results in appendix

make_accuracy_flextable <- function(tbl, path) {
  
  better_tbl <- tbl |>
    select(model, starts_with("accuracy")) |>
    mutate(across(where(is.factor), as.character)) |>
    mutate(across(-model, as.numeric))
  
  better_tbl_header <- data.frame(
    col_keys = c("model", "accuracy_overall", "accuracy_SM", "accuracy_NM", 
                 "accuracy_AM", "accuracy_no_mut", "accuracy_L", 
                 "accuracy_MS", "accuracy_no_trig"),
    line2 = c("Model", "Overall accuracy", rep("Acc. by mutation type", 4), rep("Acc. by trigger type", 3)),
    line3 = c("Model", "Overall accuracy", "SM", "NM", "AM", "None", "L", "MS", "None")
  )
  
  ft <- flextable(better_tbl, col_keys = better_tbl_header$col_keys) |>
    set_header_df(mapping = better_tbl_header, key = "col_keys") |>
    merge_v(part = "header", j = 1) |>
    merge_v(part = "header", j = 2) |>
    merge_h(part = "header", i = 1) |>
    theme_booktabs(bold_header = TRUE) |>
    align(align = "center", part = "all") |>
    vline(j = c(1, 2, 6), border = fp_border_default()) |>
    colformat_double(
      j      = c("accuracy_overall", "accuracy_SM", "accuracy_NM", "accuracy_AM",
                 "accuracy_no_mut", "accuracy_L", "accuracy_MS", "accuracy_no_trig"),
      digits = 2
    ) |>
    width(j = 1,   width = 1.4) |>
    width(j = 2,   width = 0.8) |>
    width(j = 3:6, width = 0.5) |>
    width(j = 7:9, width = 0.5) |>
    font(fontname = "Times New Roman", part = "all") |>
    border_outer(border = fp_border_default())
  
  save_as_docx(ft, path = path)
  ft
}

make_accuracy_flextable(tbl_news_accuracy_per_item, "flextable_accuracy_item_news.docx")
make_accuracy_flextable(tbl_treebank_accuracy_per_item, "flextable_accuracy_item_treebank.docx")
make_accuracy_flextable(tbl_news_expanded, "flextable_accuracy_news_expanded.docx")
make_accuracy_flextable(tbl_treebank_expanded, "flextable_accuracy_treebank_expanded.docx")
make_accuracy_flextable(tbl_news_initial, "flextable_accuracy_news_initial.docx")
make_accuracy_flextable(tbl_treebank_initial, "flextable_accuracy_treebank_initial.docx")

# Making table to compare overall accuracy by version of treebank dataset for each model

tbl_treebank_version_comparison <- tbl_treebank_accuracy_per_item |>
  select(model, accuracy_overall) |>
  rename(accuracy_overall_per_item = accuracy_overall) |>
  left_join(
    tbl_treebank_initial |> select(model, accuracy_overall) |> rename(accuracy_overall_initial = accuracy_overall),
    by = "model"
  ) |>
  left_join(
    tbl_treebank_expanded |> select(model, accuracy_overall) |> rename(accuracy_overall_expanded = accuracy_overall),
    by = "model"
  ) |>
  arrange(desc(accuracy_overall_per_item))

# Creating flextable for above

better_tbl_version <- tbl_treebank_version_comparison |>
  mutate(across(where(is.factor), as.character)) |>
  mutate(across(-model, as.numeric))

better_tbl_version_header <- data.frame(
  col_keys = c("model", "accuracy_overall_per_item", "accuracy_overall_initial", "accuracy_overall_expanded"),
  line2 = c("Model", "Accuracy per item", "Accuracy initial", "Accuracy expanded")
)

ft_version <- flextable(better_tbl_version, col_keys = better_tbl_version_header$col_keys) |>
  set_header_df(mapping = better_tbl_version_header, key = "col_keys") |>
  merge_v(part = "header", j = 1) |>
  merge_v(part = "header", j = 2) |>
  merge_h(part = "header", i = 1) |>
  theme_booktabs(bold_header = TRUE) |>
  align(align = "center", part = "all") |>
  colformat_double(
    j      = c("accuracy_overall_per_item", "accuracy_overall_initial", "accuracy_overall_expanded"),
    digits = 2
  ) |>
  width(j = 1,   width = 1.4) |>
  width(j = 2:4,   width = 0.8) |>
  font(fontname = "Times New Roman", part = "all") |>
  border_outer(border = fp_border_default())

save_as_docx(ft_version, path = "flextable_treebank_version_comparison.docx")
ft_version

# --- 3. Calculating random baseline ---

# Counting distinct incorrect_form_mut_type per item to find number of pairs per item

item_pairs <- df_treebank_expanded |>
  group_by(item) |>
  summarise(n_pairs = n_distinct(incorrect_form_mut_type), .groups = "drop") |>
  mutate(baseline = 1 / (n_pairs + 1))  # +1 to include the correct form

# Overall random baseline (weighted by number of items of each type)

random_baseline <- item_pairs |>
  summarise(
    n_1pair  = sum(n_pairs == 1),
    n_2pairs = sum(n_pairs == 2),
    n_3pairs = sum(n_pairs == 3),
    overall_baseline = mean(baseline))

# 0.31

# --- 4. Plots ---

# Making table to compare accuracy_per_item by dataset for each model

tbl_dataset_comparison_per_item <- tbl_treebank_accuracy_per_item |>
  select(model, accuracy_overall) |>
  left_join(
    tbl_news_accuracy_per_item |> select(model, accuracy_overall),
    by = "model",
    suffix = c("_treebank", "_news")
  ) |>
  arrange(desc(accuracy_overall_treebank))

# Making plot to compare mean accuracy per item by dataset for each model

tbl_comparison_long <- tbl_dataset_comparison_per_item |>
  mutate(model = factor(model, levels = tbl_dataset_comparison_per_item$model)) |>
  pivot_longer(
    cols = c(accuracy_overall_treebank, accuracy_overall_news),
    names_to  = "dataset",
    values_to = "accuracy"
  ) |>
  mutate(dataset = recode(dataset,
                          "accuracy_overall_treebank" = "Treebank (possible leaking)",
                          "accuracy_overall_news"     = "News (leaking impossible)"
  ))

ggplot(tbl_comparison_long, aes(x = model, y = accuracy, fill = dataset)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.5) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_manual(values = c("Treebank (possible leaking)" = "#a56fb3", "News (leaking impossible)" = "#76d67f")) +
  labs(x = "Model (ordered by Treebank performance)", y = "Accuracy", fill = NULL) +
  theme_minimal() +
  theme(text = element_text(size = 12, family = "sans")) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5)) +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

# From now just looking at Treebank expanded per item

# Adding "chosen mutation type" column - mutation type model has chosen out of all options (lowest NLL)

df_chosen_mut <- df_treebank_expanded |> # Making row for each option (correct + all incorrect)
  # Correct option row
  transmute(
    model, item,
    option       = "correct",
    mut_type     = as.character(mutation_type),
    nll          = correct_nll_mean
  ) |>
  bind_rows(
    # Incorrect option rows
    df_treebank_expanded |>
      transmute(
        model, item,
        option   = "incorrect",
        mut_type = as.character(incorrect_form_mut_type),
        nll      = incorrect_nll_mean
      )
  ) |>
  group_by(model, item) |>
  slice_min(nll, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(model, item, chosen_mut_type = mut_type)

# Joining onto df_treebank_item

df_treebank_item <- df_treebank_item |>
  left_join(df_chosen_mut, by = c("model", "item"))

# Plot for accuracy by training data languages

df_treebank_item <- df_treebank_item |>
  mutate(
    model_size_numeric = case_when(
      model_size == "39M"   ~ 0.039,
      model_size == "125M"  ~ 0.125,
      model_size == "1.3B"  ~ 1.3,
      model_size == "2.7B"  ~ 2.7,
      model_size == "3B"    ~ 3.0,
      model_size == "3.3B"  ~ 3.3,
      model_size == "7B"    ~ 7.0,
      model_size == "8B"    ~ 8.0,
      model_size == "9B"    ~ 9.0,
      model_size == "13B"   ~ 13.0
    ),
    size_bin = case_when(
      model_size_numeric < 1  ~ "[0, 1B)",
      model_size_numeric < 5  ~ "[1B, 5B)",
      model_size_numeric < 10 ~ "[5B, 10B)",
      TRUE                    ~ "10B+"
    ) |> factor(levels = c("[0, 1B)", "[1B, 5B)", "[5B, 10B)", "10B+"))
  )
    
    size_colours <- c("[0, 1B)" = "#4e79c4", "[1B, 5B)" = "#5ab56e", "[5B, 10B)" = "#e06fa0", "10B+" = "#9b59b6")

plot_training_data <- df_treebank_item |>
  group_by(training_data, model, size_bin) |>
  summarise(model_accuracy = mean(item_accuracy), .groups = "drop") |>
  group_by(training_data) |>
  mutate(mean_accuracy = mean(model_accuracy)) |>
  ungroup() |>
  ggplot(aes(x = reorder(training_data, -mean_accuracy))) +
  geom_point(aes(y = model_accuracy, colour = size_bin), size = 2, alpha = 0.7) +
  scale_colour_manual(values = size_colours, name = "Model size") +
  labs(
    x     = "Training data",
    y     = "Mean item accuracy"
  ) +
  scale_x_discrete(labels = c(
    "monolingual"            = "Mono\n",
    "multilingual_welsh"     = "Multi\n(Welsh)",
    "multilingual_no_celtic" = "Multi\n(no Celtic)",
    "multilingual_irish"     = "Multi\n(Irish)",
    "english"                = "English\n"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  theme(
    text             = element_text(size = 12, family = "sans"),
    axis.text.x      = element_text(vjust = 5),
    plot.title       = element_text(size = 12, hjust = 0.5),
    legend.position  = "right"
  ) +
  geom_hline(yintercept=0.31, linetype="dashed", color = "black")

ggsave("accuracy_training_data.pdf", plot = plot_training_data)

# Plotting mean accuracy by mutation type

plot_mutation_type <- df_treebank_item |>
  group_by(mutation_type) |>
  summarise(mean_accuracy = mean(item_accuracy), .groups = "drop") |>
  ggplot(aes(
    x = reorder(mutation_type, -mean_accuracy),
    y = mean_accuracy
  )) +
  geom_point(size = 5, colour = "#d4357f") +
  labs(
    x = "Mutation type",
    y = "Mean item accuracy"
  ) +
  scale_x_discrete(labels = c(
    "SM"          = "Soft",
    "NM"          = "Nasal",
    "AM"          = "Aspirate",
    "no_mutation" = "None"
  )) +
  scale_y_continuous(limits = c(0.5, 1)) +
  theme_minimal() +
  theme(
    legend.position = "none",
    text = element_text(size = 12, family = "sans"),
    axis.text.x = element_text(vjust = 1),
    plot.title = element_text(size = 12, hjust = 0.5)
  )

ggsave("accuracy_mutation_type.pdf", plot = plot_mutation_type)

# Plotting mean accuracy by trigger type

plot_trigger_type <- df_treebank_item |>
  group_by(trigger_type) |>
  summarise(mean_accuracy = mean(item_accuracy), .groups = "drop") |>
  ggplot(aes(
    x = reorder(trigger_type, -mean_accuracy),
    y = mean_accuracy
  )) +
  geom_point(size = 5, colour = "#35d4af") +
  labs(
    x = "Trigger type",
    y = "Mean item accuracy"
  ) +
  scale_x_discrete(labels = c(
    "L"           = "Lexical",
    "MS"    = "Morphosyntactic",
    "no_mutation"    = "None"
  )) +
  scale_y_continuous(limits = c(0.5, 1)) +
  theme_minimal() +
  theme(legend.position = "none") +
  theme(text = element_text(size = 12, family = "sans")) +
  theme(axis.text.x = element_text(vjust = 1)) +
  theme(plot.title = element_text(size = 12, hjust = 0.5))

ggsave("accuracy_trigger_type.pdf", plot = plot_trigger_type)

# --- 5. Fitting models ---

# Creating model for just treebank - effect of trigger_type and mutation_type on accuracy

# Changing model column to llm due to confusion with the (generalised linear mixed effects) model

df_treebank_item <- df_treebank_item |> rename(llm = model)

# Model 1
# Fitting generalised linear mixed effects model with binomial distribution and logit link function
# Fixed effects: llm, trigger_type, mutation_type (with interaction between llm and trigger_type and llm and mutation_type). Training_data colinear with llm so not including
# Random effects: sentence_id and item

mdl <- glmer(
  item_accuracy ~ llm * trigger_type + llm * mutation_type + (1 | sentence_id) + (1 | item),
  data = df_treebank_item,
  family = binomial(link = "logit"),
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6))
)

saveRDS(mdl, "model_treebank.rds")

# To load model once already fitted and saved:

# mdl <- readRDS("model_treebank.rds")

# Models in order of predicted accuracy for plots

pred_data <- predictions(
  mdl,
  newdata = datagrid(llm = levels(df_treebank_item$llm)),
  re.form = NA) |>
  as.data.frame() |>
  left_join(df_treebank_item %>% select(llm, training_data) %>% distinct(), 
            by = "llm")

model_order <- pred_data |>
  arrange(desc(estimate)) |>
  pull(llm) |>
  as.character()

# Plotting predicted accuracy by model

pred_model <- plot_predictions(mdl, condition = "llm", re.form = NA)
plot_pred_model <- pred_model +
  scale_x_discrete(limits = model_order) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        text = element_text(family = "sans", size = 12)) +
  labs(x = "Model", y = "Probability of accurate response") +
  geom_hline(yintercept = 0.31, linetype = "dashed", color = "grey")
plot_pred_model$layers[[1]]$aes_params$colour <- "#b255cf"
print(plot_pred_model)
ggsave("plot_pred_model.pdf", plot = plot_pred_model)

pred_summary <- pred_data %>%
  group_by(training_data) %>%
  summarise(
    mean_estimate = mean(estimate),
    conf.low  = mean(conf.low),
    conf.high = mean(conf.high)
  )

training_order <- c("monolingual", "multilingual_welsh", "multilingual_no_celtic", 
                    "multilingual_irish", "english")
training_labels <- c(
  "monolingual"            = "Mono",
  "multilingual_welsh"     = "Multi\n(Welsh)",
  "multilingual_no_celtic" = "Multi\n(no Celtic)",
  "multilingual_irish"     = "Multi\n(Irish)",
  "english"                = "English"
)

plot_pred_training <- ggplot(pred_summary, aes(x = training_data, y = mean_estimate)) +
  geom_pointrange(data = pred_data, aes(y = estimate, ymin = conf.low, ymax = conf.high),
                  colour = "#b255cf", size = 0.4, alpha = 1,
                  position = position_jitter(width = 0.1, height = 0, seed = 42)) +
  geom_point(colour = "grey", size = 3) +
  scale_x_discrete(limits = training_order, labels = training_labels) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 0.5),
        text = element_text(family = "sans", size = 12)) +
  labs(x = "Training data", y = "Probability of accurate response") +
  geom_hline(yintercept = 0.31, linetype = "dashed", color = "black")

print(plot_pred_training)
ggsave("plot_pred_training.pdf", plot = plot_pred_training)

# Plotting predicted accuracy according to model and trigger type

plot_pred_model_trigger <- plot_predictions(mdl, condition = list(llm = levels(df_treebank_item$llm), trigger_type = c("L", "MS")), re.form = NA) +
  scale_x_discrete(limits = model_order) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("L" = "#34ebae", "MS" = "#db3588"), labels = c(
    "L"          = "Lexical",
    "MS"          = "Morphosyntactic"),
    limits = c("L", "MS")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    text = element_text(family = "sans", size = 12),
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(x = "Model", y = "Probability of accurate response", colour = "Trigger type", fill = "Trigger type")

print(plot_pred_model_trigger)
ggsave("plot_pred_model_trigger.pdf", plot = plot_pred_model_trigger)

# Looking at effect of trigger type for each model (pairwise - MS vs L, MS vs None, L vs None)

print(comparisons(
  mdl,
  variables = list(trigger_type = "pairwise"),
  newdata = datagrid(llm = levels(df_treebank_item$llm)),
  re.form = NA
))

trigger_type_effect_model <- comparisons(
  mdl,
  variables = list(trigger_type = "pairwise"),
  newdata = datagrid(llm = levels(df_treebank_item$llm)),
  re.form = NA
) |>
  as.data.frame()

print(trigger_type_effect_model)

# - Significant negative effect for all but Bloom, mGPT and Phi 2 -
# - p < 0.001 for all but EuroLLM (p = 0.019) - 

# Taking average predictions over models to plot probability of correct answer by trigger type overall

plot_pred_overall_trigger <- avg_predictions(mdl, by = "trigger_type", re.form = NA) |>
  as.data.frame() |>
  filter(trigger_type %in% c("L", "MS")) |>
  ggplot(aes(x = trigger_type, y = estimate, ymin = conf.low, ymax = conf.high, 
             colour = trigger_type)) +
  geom_point(show.legend = FALSE, size = 3) +
  geom_errorbar(width = 0, show.legend = FALSE) +
  scale_color_manual(values = c("L" = "#34ebae", "MS" = "#db3588")) +
  scale_x_discrete(labels = c(
    "L"          = "Lexical",
    "MS"          = "Morphosyntactic"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(x = "Trigger type", y = "Probability of accurate response", colour = "Trigger type") 

print(plot_pred_overall_trigger)
ggsave("plot_pred_overall_trigger.pdf", plot = plot_pred_overall_trigger)

# Looking at effect of trigger type overall (mean)

print(avg_comparisons(mdl, variables = list(trigger_type = "pairwise"), re.form = NA))

# mean(MS)-mean(L) = -0.057. p < 0.001

# --- same as above but for mutation type ----

# Plotting predicted accuracy according to model and mutation type

plot_pred_model_mut <- predictions(
  mdl,
  newdata = datagrid(
    llm = levels(df_treebank_item$llm),
    mutation_type = c("AM", "NM", "SM")
  ),
  re.form = NA
) |>
  as.data.frame() |>
  ggplot(aes(x = llm, y = estimate, ymin = conf.low, ymax = conf.high, colour = mutation_type)) +
  geom_point(position = position_dodge(width = 0.4)) +
  geom_errorbar(width = 0, position = position_dodge(width = 0.4)) +
  scale_x_discrete(limits = model_order) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("SM" = "#9ecbf7", "NM" = "#aaf542", "AM" = "#af28f7"),
                     labels = c("SM" = "Soft", "NM" = "Nasal", "AM" = "Aspirate")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        text = element_text(family = "sans", size = 12)) +
  labs(x = "Model", y = "Probability of accurate response", colour = "Mutation type")

print(plot_pred_model_mut)
ggsave("plot_pred_model_mut.pdf", plot = plot_pred_model_mut)

# Looking at effect of mutation type for each model (pairwise)

mutation_type_effect_model <- comparisons(
  mdl,
  variables = list(mutation_type = "pairwise"),
  newdata = datagrid(llm = levels(df_treebank_item$llm)),
  re.form = NA
) |>
  as.data.frame() |>
  select(llm, contrast, estimate, conf.low, conf.high, p.value)

# Taking average predictions over models to plot probability of correct answer by mutation type overall

mut_pred <- avg_predictions(mdl, newdata = datagrid(llm = levels(df_treebank_item$llm), mutation_type = c("AM", "NM", "SM")), by = "mutation_type", re.form = NA) |>
  as.data.frame()

plot_pred_overall_mut <- mut_pred |>
  ggplot(aes(x = mutation_type, y = estimate, ymin = conf.low, ymax = conf.high,
             colour = mutation_type)) +
  geom_point(show.legend = FALSE, size = 3) +
  geom_errorbar(width = 0, show.legend = FALSE) +
  scale_color_manual(values = c("SM" = "#9ecbf7", "NM" = "#aaf542", "AM" = "#af28f7")) +
  scale_x_discrete(labels = c("SM" = "Soft", "NM" = "Nasal", "AM" = "Aspirate")) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_minimal() +
  labs(x = "Mutation type", y = "Probability of accurate response")

print(plot_pred_overall_mut)
ggsave("plot_pred_overall_mut.pdf", plot = plot_pred_overall_mut)

# Looking at effect of mutation type overall (mean)

print(avg_comparisons(mdl, variables = list(mutation_type = "pairwise"), re.form = NA))

# mean(NM)-mean(AM) = 0.13, p < 0.001
# mean(SM)-mean(AM) = 0.12, p < 0.001
# mean(SM)-mean(NM) = -0.02 but not at all significant :)

#------------------------------
# Model 2 - looking at effect of dataset (possible leaking vs no leaking)
# Fitting generalised linear mixed effects model with binomial distribution and logit link function
# Fixed effects: dataset, llm, trigger_type, mutation_type (with interaction between dataset and llm)
# Random effects: sentence_id and item

# First creating df_news_item

df_news_item <- df_news_expanded |>
  group_by(model, item) |>
  summarise(
    item_accuracy   = as.integer(all(accuracy == 1)),
    mutation_type   = first(mutation_type),
    trigger_type    = first(trigger_type),
    sentence_id     = first(sentence_id),
    training_data   = first(training_data),
    model_size      = first(model_size),
    layers          = first(layers),
    .groups = "drop"
  )

df_news_item <- df_news_item |> rename(llm = model) # renaming model column because of confusion

# Combining Treebank (possible leaking) and News (no leaking) datasets

df_item_combined <- bind_rows(
  df_treebank_item |> mutate(
    item        = paste0(item, "_treebank"), # Adding suffixes to differentiate between datasets
    sentence_id = paste0(sentence_id, "_treebank"),
    dataset     = "treebank" # Adding dataset column
  ),
  df_news_item |> mutate(
    item        = paste0(item, "_news"),
    sentence_id = paste0(sentence_id, "_news"),
    dataset     = "news"
  )
) |>
  mutate(dataset = as.factor(dataset))

# Fitting model

mdl2 <- glmer(item_accuracy ~ llm * dataset + trigger_type + mutation_type + (1 | sentence_id) + (1 | item),
      data = df_item_combined,
      family = binomial(link = "logit"),
      control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e6))
      )

saveRDS(mdl2, "model_treebank_news.rds")

# To load model once already fitted and saved:

# mdl2 <- readRDS("model_treebank_news.rds")

# Plotting predictions

plot_pred_overall_dataset <- avg_predictions(mdl2, by = "dataset", re.form = NA) |>
  as.data.frame() |>
  ggplot(aes(x = dataset, y = estimate, ymin = conf.low, ymax = conf.high, 
             colour = dataset)) +
  geom_point(show.legend = FALSE, size = 3) +
  geom_errorbar(width = 0, show.legend = FALSE) +
  scale_color_manual(values = c("treebank" = "#a56fb3", "news" = "#76d67f")) +
  scale_y_continuous(limits = c(0, 1)) + 
  scale_x_discrete(labels = c(
    "news" = "News (leaking impossible)",
    "treebank" = "Treebank (leaking possible)"
    ))  +
  theme_minimal() +
  labs(x = "Dataset", y = "Probability of accurate response", colour = "Dataset")

print(plot_pred_overall_dataset)
ggsave("plot_pred_overall_dataset.pdf", plot = plot_pred_overall_dataset)

# Looking at effect of dataset overall (mean)

print(avg_comparisons(mdl2, variables = list(dataset = "pairwise"), re.form = NA))

# B 0.04, significant (p = 0.03)

plot_pred_model_dataset <- plot_predictions(mdl2, condition = c("llm", "dataset"), re.form = NA) +
  scale_x_discrete(limits = model_order) +
  scale_y_continuous(limits = c(0, 1)) +
  scale_color_manual(values = c("treebank" = "#a56fb3", "news" = "#76d67f"), labels = c(
      "news" = "News (leaking impossible)",
      "treebank" = "Treebank (leaking possible)"
    )) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
        text = element_text(family = "sans", size = 12),
        plot.title = element_text(hjust = 0.5)
  ) +
  labs(x = "Model", y = "Probability of accurate response", colour = "Dataset", fill = "Dataset")

print(plot_pred_model_dataset)
ggsave("plot_pred_model_dataset.pdf", plot = plot_pred_model_dataset)

# Looking at effect of dataset for each model

dataset_effect_model <- comparisons(
  mdl2,
  variables = list(dataset = "pairwise"),
  newdata = datagrid(llm = levels(df_treebank_item$llm)),
  re.form = NA
) |>
  as.data.frame() |>
  select(llm, contrast, estimate, conf.low, conf.high, p.value)

# Only significant contrasts:
# Bloom 0.14 (Treebank higher), p < 0.001
# Goldfish 100mb -0.02 (News higher), p = 0.04
# Goldfish 5mb 0.07 (Treebank higher), p = 0.04
# EuroLLM 0.12 (Treebank higher), p = 0.01

saveRDS(mdl2, "model_treebank_news.rds")

# --- 6. Qualitative analysis - hardest/easiest sentences ---

df_hardest_sentences <- df_treebank_item |>
  group_by(item) |>
  summarise(
    n_models         = n(),
    n_correct        = sum(item_accuracy),
    n_wrong          = n_models - n_correct,
    prop_wrong       = n_wrong / n_models,
    mutation_type    = first(mutation_type),
    trigger_type     = first(trigger_type),
    sentence_id      = first(sentence_id),
    .groups          = "drop"
  ) |>
  arrange(desc(n_wrong)) |>
  slice_head(n = 300) |>
  left_join(
    df_treebank_expanded |>
      select(item, correct_form, correct_sentence, specific_trigger) |>
      distinct(item, .keep_all = TRUE),
    by = "item"
  )


# Getting which models got each item wrong and what they chose

wrong_models <- df_treebank_expanded |>
  filter(accuracy == 0) |>
  group_by(item, model) |>
  slice_min(incorrect_nll_mean, n = 1, with_ties = FALSE) |>
  ungroup() |>
  group_by(item) |>
  summarise(
    models_wrong = paste(model, collapse = ", "),
    incorrect_forms_chosen = paste(incorrect_form, collapse = ", "),
    incorrect_mut_types_chosen = paste(incorrect_form_mut_type, collapse = ", "),
    .groups = "drop"
  )

right_models <- df_treebank_item |>
  filter(item_accuracy == 1) |>
  group_by(item) |>
  summarise(
    models_right = paste(llm, collapse = ", "),
    .groups = "drop"
  )

df_hardest_sentences <- df_hardest_sentences |>
  left_join(wrong_models, by = "item") |>
  left_join(right_models, by = "item")

write.csv(df_hardest_sentences, "hardest_sentences.csv")

df_hardest_sentences |> count(mutation_type, sort = TRUE)
df_hardest_sentences |> count(trigger_type, sort = TRUE)
df_hardest_sentences |> count(specific_trigger, sort = TRUE)

df_hardest_sentences |> count(mutation_type, sort = TRUE) |> mutate(pct = round(n / nrow(df_hardest_sentences) * 100, 2))
df_hardest_sentences |> count(trigger_type, sort = TRUE) |> mutate(pct = round(n / nrow(df_hardest_sentences) * 100, 2))
df_hardest_sentences |> count(specific_trigger, sort = TRUE) |> mutate(pct = round(n / nrow(df_hardest_sentences) * 100, 2))

# 16 items all 15 models incorrect

# Examining easiest sentences

df_easiest_sentences <- df_treebank_item |>
  group_by(item) |>
  summarise(
    n_models        = n(),
    n_correct       = sum(item_accuracy),
    n_wrong         = n_models - n_correct,
    prop_wrong      = n_wrong / n_models,
    mutation_type   = first(mutation_type),
    trigger_type    = first(trigger_type),
    sentence_id     = first(sentence_id),
    .groups = "drop"
  ) |>
  arrange(desc(n_correct)) |>
  slice_head(n = 1000) |>
  left_join(
    df_treebank_expanded |>
      select(item, correct_form, correct_sentence, specific_trigger) |>
      distinct(item, .keep_all = TRUE),
    by = "item"
  )

# 631 items all 15 models correct

# --- 7. Confusion matrices ---

# Creating confusion matrices, first for all models then by model

mut_levels <- c("SM", "NM", "AM", "no_mutation")

confusion_all <- df_treebank_item |>
  count(mutation_type, chosen_mut_type) |>
  rename(correct = mutation_type, chosen = chosen_mut_type)

confusion_per_model <- df_treebank_item |>
  count(llm, mutation_type, chosen_mut_type) |>
  rename(correct = mutation_type, chosen = chosen_mut_type)

# For models acc > 80%

top6_models <- c(
  "Goldfish 1000mb",
  "Goldfish 100mb",
  "BritLLM 3B",
  "LLaMAX3 8B",
  "Goldfish 10mb",
  "TransWebLLM"
)

# As matrices not long format

confusion_all_matrix <- confusion_all |>
  mutate(
    correct = factor(correct, levels = mut_levels),
    chosen  = factor(chosen,  levels = mut_levels)
  ) |>
  pivot_wider(names_from = chosen, values_from = n, values_fill = 0) |>
  arrange(correct) |>
  column_to_rownames("correct") |>
  select(all_of(mut_levels))

confusion_model_matrix <- confusion_per_model |>
  group_by(llm) |>
  group_split() |>
  setNames(levels(df_treebank_item$llm)) |>
  lapply(\(df)
         df |>
           select(-llm) |>
           mutate(
             correct = factor(correct, levels = mut_levels),
             chosen  = factor(chosen,  levels = mut_levels)
           ) |>
           pivot_wider(names_from = chosen, values_from = n, values_fill = 0) |>
           arrange(correct) |>
           column_to_rownames("correct") |>
           select(all_of(mut_levels))
  )

confusion_top6_matrix <- confusion_per_model |>
  filter(llm %in% top6_models) |>
  group_by(correct, chosen) |>
  summarise(n = sum(n), .groups = "drop") |>
  mutate(correct = factor(correct, levels = mut_levels),
    chosen  = factor(chosen,  levels = mut_levels)) |>
  pivot_wider(names_from = chosen, values_from = n, values_fill = 0) |>
  arrange(correct) |>
  column_to_rownames("correct") |>
  select(all_of(mut_levels))

confusion_top6_pct <- confusion_top6_matrix |>
  as.data.frame() |>
  rownames_to_column("correct") |>
  mutate(across(all_of(mut_levels), \(x) round(x / rowSums(across(all_of(mut_levels))) * 100, 2))) |>
  column_to_rownames("correct")

confusion_all_pct <- confusion_all_matrix |>
  as.data.frame() |>
  rownames_to_column("correct") |>
  mutate(across(all_of(mut_levels), \(x) round(x / rowSums(across(all_of(mut_levels))) * 100, 2))) |>
  column_to_rownames("correct")

print(confusion_all_matrix)
print(confusion_model_matrix)
print(confusion_top6_matrix)
print(confusion_top6_pct)
print(confusion_all_pct)

# --- 8. LOGIT LENS analysis (on Simple Treebank) ---

df_goldfish1000_ll <- read.csv("https://raw.githubusercontent.com/bethancunningham/tfm/main/logit_lens_results/ll_data_goldfish1000.csv")

df_britllm_ll <- read.csv("https://raw.githubusercontent.com/bethancunningham/tfm/main/logit_lens_results/ll_data_britllm.csv")

# Function to parse lists of NLLs to matrix

parse_to_matrix <- function(x) {
  x |>
    gsub("\\[|\\]", "", x = _) |>
    strsplit(",") |>
    sapply(as.numeric) |>
    t()
}

# Function to plot logit lens results (mean delta) by trigger

plot_ll_by_trigger <- function(df, model_name) {
  
  mat_correct   <- parse_to_matrix(df$layer_nll_correct)
  mat_incorrect <- parse_to_matrix(df$layer_nll_incorrect)
  df$layer_delta <- mat_incorrect - mat_correct
  
  plot_df <- df |>
    filter(trigger_type %in% c("L", "MS", "no_mutation")) |>
    group_by(trigger_type) |>
    group_modify(~{
      means <- colMeans(.x$layer_delta, na.rm = TRUE)
      sds   <- apply(.x$layer_delta, 2, sd, na.rm = TRUE)
      data.frame(layer = seq_along(means) - 1, mean = means, sd = sds)
    })
  
  p <- ggplot(plot_df, aes(x = layer, y = mean, colour = trigger_type, fill = trigger_type)) +
    geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd), alpha = 0.15, colour = NA) +
    geom_line() +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    scale_colour_manual(values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"),
                        labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None")) +
    scale_fill_manual(values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"),
                      labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None")) +
    scale_x_continuous(breaks = seq(0, max(plot_df$layer), by = 2), expand = c(0, 0)) +
    coord_cartesian(ylim = c(-0.8, 0.8)) +
    labs(x = "Layer", y = "Mean NLL delta (incorrect − correct)",
         colour = "Trigger type", fill = "Trigger type") +
    theme_minimal() +
    theme(text = element_text(family = "sans", size = 12))
  
  ggsave(paste0(model_name, "_ll_by_trigger.pdf"), plot = p, width = 7, height = 4)
  
  return(p)
}

plot_ll_by_trigger(df_britllm_ll, model_name = "BritLLM 3B")
plot_ll_by_trigger(df_goldfish1000_ll,  model_name = "Goldfish 1000mb")

# Function to plot logit lens results (mean delta) by specific trigger

plot_ll_by_specific_trigger <- function(df) {
  
  mat_correct   <- parse_to_matrix(df$layer_nll_correct)
  mat_incorrect <- parse_to_matrix(df$layer_nll_incorrect)
  df$layer_delta <- mat_incorrect - mat_correct
  
  plot_df <- df |>
    filter(trigger_type == "MS") |>
    group_by(specific_trigger) |>
    group_modify(~{
      means <- colMeans(.x$layer_delta, na.rm = TRUE)
      sds   <- apply(.x$layer_delta, 2, sd, na.rm = TRUE)
      data.frame(layer = seq_along(means) - 1, mean = means, sd = sds)
    })
  
  ggplot(plot_df, aes(x = layer, y = mean)) +
    geom_ribbon(aes(ymin = mean - sd, ymax = mean + sd), alpha = 0.15, fill = "steelblue", colour = NA) +
    geom_line(colour = "steelblue") +
    geom_hline(yintercept = 0, linetype = "dashed", colour = "grey40") +
    facet_wrap(~specific_trigger) +
    labs(x = "Layer", y = "Mean NLL delta (incorrect − correct)") +
    theme_minimal() +
    theme(text = element_text(family = "sans", size = 12))
}

plot_ll_by_specific_trigger(df_britllm_ll)
plot_ll_by_specific_trigger(df_goldfish1000_ll)

# Converting ll dfs to long format to fit models

# Function to convert ll df to long format

prepare_ll_df <- function(df) {
  long_df <- df %>%
    mutate(
      nll_correct = map(layer_nll_correct, ~as.numeric(str_remove_all(.x, "\\[|\\]") %>% str_split(",") %>% .[[1]])),
      nll_incorrect = map(layer_nll_incorrect, ~as.numeric(str_remove_all(.x, "\\[|\\]") %>% str_split(",") %>% .[[1]])),
      layer = map(nll_correct, ~seq_along(.x) - 1L)
    ) %>%
    unnest(c(nll_correct, nll_incorrect, layer)) %>%
    mutate(delta = nll_incorrect - nll_correct)
  }

# Applying function

df_goldfish1000_ll_long <- prepare_ll_df(df_goldfish1000_ll)
df_britllm_ll_long <- prepare_ll_df(df_britllm_ll)

# Renaming index column

df_goldfish1000_ll_long <- df_goldfish1000_ll_long %>%
  rename(item = index)
df_britllm_ll_long <- df_britllm_ll_long %>%
  rename(item = index)

# Making certain columns factors

df_goldfish1000_ll_long <- df_goldfish1000_ll_long %>%
  mutate(
    delta = as.numeric(delta),
    across(c(trigger_type, layer, mutation_type, sentence_id, item), factor)
  )

df_britllm_ll_long <- df_britllm_ll_long %>%
  mutate(
    delta = as.numeric(delta),
    across(c(trigger_type, layer, mutation_type, sentence_id, item), factor)
  )

# Fitting models

mdl_ll_gf <- lmer(
  delta ~ trigger_type * layer + mutation_type + (1 | sentence_id) + (1 | item), 
  data = df_goldfish1000_ll_long,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

mdl_ll_britllm <- lmer(
  delta ~ trigger_type * layer + mutation_type + (1 | sentence_id) + (1 | item), 
  data = df_britllm_ll_long,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
)

# Saving models

saveRDS(mdl_ll_gf, file = "mdl_ll_gf.rds")
saveRDS(mdl_ll_britllm, file = "mdl_ll_britllm.rds")

# Plotting predictions

plot_data_gf <- predictions(mdl_ll_gf, by = c("layer", "trigger_type"), re.form = NA)

ggplot(plot_data_gf, aes(x = layer, y = estimate, color = trigger_type, group = trigger_type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = trigger_type), 
              alpha = 0.2, colour = NA) +
  theme_minimal() +
  labs(
    x = "Layer",
    y = "Predicted delta",
    color = "Trigger type"
  ) +
  scale_colour_manual(name = "Trigger type", values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"), labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None")) +
  scale_fill_manual(name = "Trigger type", values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"), labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None"))


plot_data_brit <- predictions(mdl_ll_britllm, by = c("layer", "trigger_type"), re.form = NA)

ggplot(plot_data_brit, aes(x = layer, y = estimate, color = trigger_type, group = trigger_type)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 3) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high, fill = trigger_type), 
              alpha = 0.2, colour = NA) +
  theme_minimal() +
  labs(
    x = "Layer",
    y = "Predicted delta",
    color = "Trigger type"
  ) +
    scale_colour_manual(name = "Trigger type", values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"), labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None")) +
  scale_fill_manual(name = "Trigger type", values = c("L" = "#34ebae", "MS" = "#db3588", "no_mutation" = "#ed9b3b"), labels = c("L" = "Lexical", "MS" = "Morphosyntactic", "no_mutation" = "None"))

# Getting pairwise comparisons at each layer

cmp_gf <- comparisons(
  mdl_ll_gf,
  variables = list(trigger_type = c("L", "MS")),
  by = "layer",
  newdata = subset(df_goldfish1000_ll_long, trigger_type %in% c("L", "MS")),
  p_adjust = "fdr"
)

cmp_britllm <- comparisons(
  mdl_ll_britllm,
  variables = list(trigger_type = c("L", "MS")),
  by = "layer",
  newdata = subset(df_britllm_ll_long, trigger_type %in% c("L", "MS")),
  p_adjust = "fdr"
)

# Making tables from the above

# Goldfish

cmp_gf_tbl <- cmp_gf %>%
  select(layer, estimate, std.error, statistic, p.value) %>%
  mutate(
    sig = p.value < 0.05,
    across(c(estimate, std.error, statistic), ~round(., 3)),
    p.value = ifelse(p.value < 0.001, "<0.001", 
                     formatC(p.value, digits = 3, format = "f"))
    )

ft1_gf <- flextable(cmp_gf_tbl %>% select(-sig)) %>%
  set_header_labels(layer = "Layer", estimate = "Estimate", 
                    std.error = "SE", statistic = "z", p.value = "p") %>%
  bold(i = which(cmp_gf_tbl$sig), part = "body") %>%
  autofit()

save_as_image(ft1_gf, path = "table_gf.png", res = 300)

# BritLLM

cmp_britllm_tbl <- cmp_britllm %>%
  select(layer, estimate, std.error, statistic, p.value) %>%
  mutate(
    sig = p.value < 0.05,
    across(c(estimate, std.error, statistic), ~round(., 3)),
    p.value = ifelse(p.value < 0.001, "<0.001", 
                     formatC(p.value, digits = 3, format = "f"))
    )

ft1_britllm <- flextable(cmp_britllm_tbl %>% select(-sig)) %>%
  set_header_labels(layer = "Layer", estimate = "Estimate",
                    std.error = "SE", statistic = "z", p.value = "p") %>%
  bold(i = which(cmp_britllm_tbl$sig), part = "body") %>%
  autofit()

save_as_image(ft1_britllm, path = "table_britllm.png", res = 300)

