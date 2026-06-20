import requests
import conllu
import pandas as pd
import re
from math import log

# Creating dictionary with mutation rules

MUTATION_RULES = {
    "c":  {"SM": ("g",   1), "NM": ("ngh", 1), "AM": ("ch",  1)},
    "p":  {"SM": ("b",   1), "NM": ("mh",  1), "AM": ("ph",  1)},
    "t":  {"SM": ("d",   1), "NM": ("nh",  1), "AM": ("th",  1)},
    "g":  {"SM": ("",    1), "NM": ("ng",  1)},
    "b":  {"SM": ("f",   1), "NM": ("m",   1)},
    "d":  {"SM": ("dd",  1), "NM": ("n",   1)},
    "m":  {"SM": ("f",   1)},
    "ll": {"SM": ("l",   2)},
    "rh": {"SM": ("r",   2)},
}

url = "https://raw.githubusercontent.com/bethancunningham/tfm/main/datasets/sentences_news_articles_semicolon.csv"
df_small = pd.read_csv(url, sep=";")
print(df_small.head(10))

# Adding token_id and lemma column to match other df and use the same code

def get_token_info(row):
    """Function to find correct_form in sentence tokens and returning index (token_id) and the previous word."""
    tokens = row["correct_sentence"].split()
    correct = row["correct_form"]
    
    for i, token in enumerate(tokens):
        clean_token = token.strip(".,!?;:\"'""''()-")
        if clean_token == correct:
            token_id = i + 1
            prev_word = tokens[i - 1].strip(".,!?;:\"'""''()-") if i > 0 else None
            return token_id, prev_word
    
    return None, None

def unmutate(row):
    """Function to take mutated word in row and return unmutated (root) word according to type of mutation."""
    word = row["correct_form"].lower()
    mut = row["mutation_type"]
    lemma = row["lemma"].lower()

    if mut == "SM":
        if lemma.startswith("g") or word in {"orau", "well"}: return "g" + word # accounting for irregular instances - gwell 'better' gorau 'best', lemma = da 'good'
        elif lemma.startswith("m"):  return "m" + word[1:] # to differentiate between m to f and b to f
        elif word.startswith("f"):   return "b" + word[1:]
        elif word.startswith("dd"):  return "d" + word[2:]
        elif word.startswith("b"):   return "p" + word[1:]
        elif word.startswith("d"):   return "t" + word[1:]
        elif word.startswith("g"):   return "c" + word[1:]
        elif word.startswith("l"):   return "ll" + word[1:]
        elif word.startswith("r"):   return "rh" + word[1:]
        else: return None

    if mut == "AM":
        if word.startswith("ch"):    return "c" + word[2:]
        elif word.startswith("th"):  return "t" + word[2:]
        elif word.startswith("ph"):  return "p" + word[2:]
        else: return None

    if mut == "NM":
        if word.startswith("ngh"):   return "c" + word[3:]
        elif word.startswith("mh"):  return "p" + word[2:]
        elif word.startswith("nh"):  return "t" + word[2:]
        elif word.startswith("ng"):  return "g" + word[2:]
        elif word.startswith("m"):   return "b" + word[1:]
        elif word.startswith("n"):   return "d" + word[1:]
        else: return None

    return None

def apply_mutations(root, exclude=None):
    """Function to return {mut_type: mutated_form} for all applicable mutations of root word."""
    w = root.lower()
    cap = lambda s: s[0].upper() + s[1:] if root[0].isupper() else s
    key = next((k for k in ("ll", "rh", "c", "p", "t", "g", "b", "d", "m") if w.startswith(k)), None)
    if key is None:
        return {}
    return {
        mut: cap(prefix + w[strip:]) # strip removes first letter or two depending on dictionary key
        for mut, (prefix, strip) in MUTATION_RULES[key].items()
        if (prefix + w[strip:]) and mut != exclude # skips mutations already in df, e.g. if SM correct only provides NM and AM, if applicable
    }

def make_incorrect_sentence(row):
    """Function to replace correct_form in sentence with incorrect_form to create incorrect sentence."""
    sentence = row["correct_sentence"]
    correct = row["correct_form"]
    incorrect = row["incorrect_form"]
    prev = row["prev_word"]
    token_id = row["token_id"]

    contexts = []
    if prev and isinstance(prev, str):  # checking it's a string because of previous error
        contexts.append(prev)

    def find_matches(context):
        """Function to search for word plus its context to ensure the correct occurrence of the word. Returns matches."""
        pattern = re.escape(context) + r"\s[\u201c\u2018]?\d*" + re.escape(correct) + r"(?!\w)"
        return [(m.start() + m.group().index(correct), m.start() + m.group().index(correct) + len(correct))
                for m in re.finditer(pattern, sentence)]

    all_matches = []
    for ctxt in contexts:
        all_matches = find_matches(ctxt)
        if all_matches:
            break

    if not all_matches:
        all_matches = [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(correct) + r"(?!\w)", sentence)]  # Finding start and end position of every occurrence of the word on its own

    if not all_matches:
        print(f"Warning: 0 matches found for '{correct}' in: {sentence}")
        return sentence

    if len(all_matches) == 1:
        start, end = all_matches[0]
    else:
        approx_pos = len(" ".join(sentence.split()[:token_id - 1]))
        start, end = min(all_matches, key=lambda m: abs(m[0] - approx_pos)) # If more than one match - find closest to the approx position of the word in the sequence of tokens

    return sentence[:start] + incorrect + sentence[end:]

def check_one_diff(row):
    """Function to check number of different tokens between correct and incorrect sentences - should be 1."""
    orig   = row["correct_sentence"].split()
    incorr = row["incorrect_sentence"].split()
    if len(orig) != len(incorr):
        return "different_length"
    return sum(a != b for a, b in zip(orig, incorr))


# Applying functions to df_small

# Getting token_id and prev_word

df_small[["token_id", "prev_word"]] = df_small.apply(
    lambda row: pd.Series(get_token_info(row)), axis=1
)

# Mutated rows: unmutating to get incorrect_form

mutated_mask = df_small["mutation_type"].isin(["SM", "NM", "AM"]) # selecting only rows where correct_form is mutated
df_small.loc[mutated_mask, "incorrect_form"] = df_small[mutated_mask].apply(unmutate, axis=1)

# Restoring capitalisation for mutated rows

df_small.loc[mutated_mask, "incorrect_form"] = df_small[mutated_mask].apply(
    lambda row: row["incorrect_form"][0].upper() + row["incorrect_form"][1:]
    if row["incorrect_form"] and row["correct_form"][0].isupper() else row["incorrect_form"], axis=1
)

# Unmutated rows: mutating to get incorrect_form

unmutated_mask = df_small["mutation_type"].isin(["no_SM", "no_NM", "no_AM"])
df_small.loc[unmutated_mask, "incorrect_form"] = df_small[unmutated_mask].apply(
    lambda row: apply_mutations(row["correct_form"]).get(row["mutation_type"].replace("no_", "")),
    axis=1
)

# Creating incorrect sentences

df_small["incorrect_sentence"] = df_small.apply(make_incorrect_sentence, axis=1)

# Sanity check - checking only 1 difference between correct and incorrect sentences

df_small["num_diffs"] = df_small.apply(check_one_diff, axis=1)
print(df_small["num_diffs"].value_counts())

# Checking counts in initial df and exporting

print("Mutation type counts in expanded df:\n", df_small["mutation_type"].value_counts())
print("Trigger type counts in initial df:\n", df_small["trigger_type"].value_counts())
print("Rows in expanded df:\n", len(df_small))

print("Exporting small df (initial version, 1 pair per item) to csv (name: initial_news_dataset.csv)")
df_small.to_csv("initial_news_dataset.csv", index=False)

# Calculating average sentence length and type/token ratio for news dataset 
# Adding sentence_tokens column

def tokenise_welsh(text): # simple tokeniser by space and apostrophe adequate for this purpose
    # First split on whitespace
    words = str(text).lower().split()
    tokens = []
    for word in words:
        # Splitting off clitics and contractions like 'r, 'n, 'm, 'th, 'ch, 'i
        parts = re.split(r"(?<=\w)('r|'n|'m|'th|'ch|'i|'w)\b", word)
        tokens.extend([p for p in parts if p])
    return tokens

df_small["sentence_tokens"] = df_small["correct_sentence"].apply(tokenise_welsh)

df_small_tokens = [t for tokens in df_small.drop_duplicates("sent_id")["sentence_tokens"] for t in tokens]
n_small_sents   = df_small["sent_id"].nunique()

print("\nNews df sentences")
print(f"Average sentence length: {len(df_small_tokens) / n_small_sents}")
print(f"Type/token ratio: {len(set(df_small_tokens)) / len(df_small_tokens)}")
print(f"Herdan's C: {log(len(set(df_small_tokens))) / log(len(df_small_tokens))}\n-------\n")

# Generating extra rows with all possible incorrect_form options

def expand_rows(row) :
    """Function to yield one dict (row) per alternative mutation form, plus the original row."""
    base = row.to_dict() # Making dictionary from row
    mut = row["mutation_type"]
    root = row["correct_form"] if mut.startswith("no_") else row["incorrect_form"]
    exclude = mut.replace("no_", "") if mut.startswith("no_") else mut # Don't repeat already provided mutation forms

    # Getting original pair
    original_incorrect_type = exclude if mut.startswith("no_") else "none"
    yield {**base, "incorrect_form_mut_type": original_incorrect_type} # Original row with one extra column: incorrect_form_mut_type

    # Additional pairs for each other possible incorrect mutations
    for alt_mut, alt_form in apply_mutations(root, exclude=exclude).items():
        yield {**base, # Taking copy of original row dict and replacing incorrect_form etc.
               "incorrect_form": alt_form,
               "incorrect_form_mut_type": alt_mut,
               "incorrect_sentence": make_incorrect_sentence({**base, "incorrect_form": alt_form})}

# Adding rows for all possible incorrect forms

df_small_expanded = pd.DataFrame(
    r for _, row in df_small.iterrows() for r in expand_rows(row)
).reset_index(drop=True)

df_small_expanded["num_diffs"] = df_small_expanded.apply(check_one_diff, axis=1)
print(df_small_expanded["num_diffs"].value_counts()) # checking only 1 diff between correct and incorrect sentence in expanded df - yes

# Checking mutation type counts in expanded df

print("Mutation type counts in expanded df:\n", df_small_expanded["mutation_type"].value_counts())
print("Trigger type counts in initial df:\n", df_small_expanded["trigger_type"].value_counts())
print("Rows in expanded df:\n", len(df_small_expanded))
print("Exporting expanded small df to csv (name: expanded_news_dataset.csv)")
df_small_expanded.to_csv("expanded_news_dataset.csv", index=True)
print("-------\n")
