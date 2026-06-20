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

# Getting list of feminine singular nouns from Welsh lexicon (needed later)

url_femsingnouns = "https://raw.githubusercontent.com/bethancunningham/tfm/main/source_files/feminine_singular_nouns_full_mutations.txt"
request_0 = requests.get(url_femsingnouns)
femsingnouns_file = request_0.text
femsingnouns_lexicon = []
for word in femsingnouns_file.splitlines() :
    femsingnouns_lexicon.append(word.strip().lower())
    
# Reading in Welsh UD Treebank info - split into train, dev and test

url_train = "https://raw.githubusercontent.com/bethancunningham/tfm/main/source_files/treebank_train.conllu"
request_1 = requests.get(url_train)
train_file = request_1.text

url_dev = "https://raw.githubusercontent.com/bethancunningham/tfm/main/source_files/treebank_dev.conllu"
request_2 = requests.get(url_dev)
dev_file = request_2.text

url_test = "https://raw.githubusercontent.com/bethancunningham/tfm/main/source_files/treebank_test.conllu"
request_3 = requests.get(url_test)
test_file = request_3.text

# Merging 3 files

files = [train_file, dev_file, test_file]

merged_tokenlists = []
for file in files:
    for tokenlist in conllu.parse(file):
        merged_tokenlists.append(tokenlist)

print("-------\n", "Total number of sentences initially:", len(merged_tokenlists), "\n-------")

# Creating dataframe with data from treebank

data = []

for sentence in merged_tokenlists:
  sent_id = sentence.metadata.get("sent_id")
  text = sentence.metadata.get("text")

# Iterating over tokens, finding ones with mutation and adding them and necessary accompanying data to df

  for i, token in enumerate(sentence) :
      feats = token.get("feats")

      if feats and "Mutation" in feats :
          prev_token = sentence[i-1] if i > 0 else None
          head_id = token["head"]
          head_word = sentence[head_id - 1] if head_id and head_id > 0 else None

          data.append({
                "sent_id": sent_id,
                "sentence": text,
                "sentence_tokens": [t["form"] for t in sentence],
                "sentence_en": sentence.metadata.get("text_en"),
                "correct_form": token["form"],
                "lemma": token["lemma"],
                "mutation_type": feats["Mutation"],
                "token_id": token["id"],
                "pos": token["upos"],
                "dep_rel": token["deprel"],
                "features": token.get("feats"),
                "prev_word": sentence[i-1]["form"] if i > 0 else None,
                "prev_lemma": sentence[i-1]["lemma"] if i > 0 else None,
                "prev_pos": prev_token["upos"] if prev_token else None,
                "prev_feats": prev_token.get("feats") if prev_token else None,
                "before_previous_word": sentence[i-2]["form"] if i > 1 else None,
                "head_id": token["head"],
                "head_word": head_word["form"] if head_word else None,
                "head_lemma": head_word["lemma"] if head_word else None,
                "head_pos": head_word["upos"] if head_word else None,
                "head_feats": head_word.get("feats") if head_word else None,
                })

df = pd.DataFrame(data)

print("No. instances of mutation initially:", len(df), "\n-------")

df = df[~df["correct_form"].str.startswith("h")] # not including h-prothesis

# Adding extra column with unmutated word

def unmutate(row) :
    """Function to take mutated word in row and return unmutated (root) word according to type of mutation"""
    word = row["correct_form"].lower()
    mut = row["mutation_type"]
    lemma = row["lemma"].lower()

    if mut == "SM":
        if lemma.lower().startswith("g") or word == "orau" or word == "well" : return "g" + word # accounting for irregular instances - gwell 'better' gorau 'best', lemma = da 'good'
        elif lemma.lower().startswith("m") :    return "m" + word[1:] # to differentiate between m to f and b to f
        elif word.startswith("f") :    return "b" + word[1:]
        elif word.startswith("dd") :   return "d" + word[2:]
        elif word.startswith("b") :    return "p" + word[1:]
        elif word.startswith("d") :    return "t" + word[1:]
        elif word.startswith("g") :    return "c" + word[1:]
        elif word.startswith("l") :    return "ll" + word[1:]
        elif word.startswith("r") :    return "rh" + word[1:]
        else : return None

    if mut == "AM":
        if word.startswith("ch") :     return "c" + word[2:]
        elif word.startswith("th") :   return "t" + word[2:]
        elif word.startswith("ph") :   return "p" + word[2:]
        else: return None

    if mut == "NM":
        if word.startswith("ngh") :    return "c" + word[3:]
        elif word.startswith("mh") :   return "p" + word[2:]
        elif word.startswith("nh") :   return "t" + word[2:]
        elif word.startswith("ng") :   return "g" + word[2:]
        elif word.startswith("m") :    return "b" + word[1:]
        elif word.startswith("n") :    return "d" + word[1:]
        else: return None

df["incorrect_form"] = df.apply(unmutate, axis=1)

# Checking None values
# print(df["incorrect_form"].isna().sum())
# print(df[["correct_form", "lemma"]][df["incorrect_form"].isna()].head(13))

# Removing rows with no unmutated version as these are errors - not really cases of mutation 

df = df[df["incorrect_form"].notna()]

# Putting capital letter back if mutated word starts with capital letter

df["incorrect_form"] = df.apply(lambda row: row["incorrect_form"][0].upper() + row["incorrect_form"][1:] if row["correct_form"][0].isupper() else row["incorrect_form"], axis=1)

print("Total no. of sentences after removal of h-prothesis and errors", df["sent_id"].nunique())
print("No. instances of mutation after removal of h-prothesis and errors:", len(df), "\n-------")

# Creating lists of lexical triggers and lexicalised mutations

lexical_triggers_AM = ["a", "â", "chwe", "efo", "ei", "gyda", "na", "ni", "oni", "tra", "tri", "tua", "'i", "'w"]
lexical_triggers_SM = ["a", "ail", "am", "ambell", "amryw", "ar", "at", "ba", "bod", "cryn", "cwbl", "cyfryw", "cymharol", "cyn", "dacw", "dan", "dau", "ddau", "dros", "drwy", "dwy", "ddwy", "dy", "dyma", "dyna", "ei", "fod", "gan", "go", "gwbl", "gweddol", "heb", "holl", "hollol", "hyd", "hynod", "i", "lled", "mha", "mor", "na", "naill", "neu", "newydd", "ni", "o", "oll", "oni", "pa", "pan", "pedwar", "pedwaredd", "po", "pur", "pwy", "rhy", "rhyw", "saith", "sir", "tair", "tan", "tros", "trwy", "trydedd", "unrhyw", "weled", "wrth", "wyth", "ychydig", "ynteu", "'th", "'i", "'w", "'na", "'ma"]
lexicalised_mut = ["ddoe", "bynnag", "beth", "fenni", "fodd", "draw", "gilydd", "be", "faint", "drachefn", "bontnewydd", "dan"]

# Creating function to classify mutation trigger

def classify_trigger(row) :
    """Function to take row and return mutation trigger type - morphosyntactic, lexical or lexicalised"""
    mut = row["mutation_type"]
    prev = row["prev_word"]
    prev_lemma = row["prev_lemma"]
    prev_pos = row["prev_pos"]
    form = row["correct_form"]
    before_previous_word = row["before_previous_word"]

    if mut == "NM" :
        return "L"

    if mut == "AM" :
        if prev and prev.lower() in lexical_triggers_AM:
            return "L"
        else:
            return "MS"

    if mut == "SM" :
        if form.lower() in lexicalised_mut :
            return "LEXICALISED"
        elif prev and prev.lower() in lexical_triggers_SM:
            return "L"
        elif prev and prev.lower() in {"fe", "mi"} and prev_pos == "PART" : # checking for fe and mi as particles - not pronouns
            return "L"
        elif prev and prev_lemma == "bod" : # checking for forms of bod 'be'
            return "L"
        elif prev and prev == "\"" or prev == "-" :
            if before_previous_word and before_previous_word in lexical_triggers_SM :
                return "L"
            else:
                return "MS"
        elif prev and prev_lemma in lexical_triggers_SM : # to catch row with "chan" as prev word in error - actually "gan"
            return "L"
        else:
            return "MS"

    return None

# Classifying triggers (lexical/morphosyntactic) and viewing counts

df["trigger_type"] = df.apply(classify_trigger, axis=1)

print("Mutation type and trigger type counts:", df.groupby(["mutation_type", "trigger_type"]).size(), "\n-------")

# Removing cases of lexicalised mutation

df = df[df["trigger_type"] != "LEXICALISED"]

print("Total no. of sentences after removal of h-prothesis and errors and lexicalised mutations", df["sent_id"].nunique())
print("No. instances of mutation after removal of h-prothesis and errors and lexicalised mutations:", len(df), "\n-------")

# Function for identifying specific trigger (e.g. direct object, vocative, singular feminine noun preceded by article)

adverbs_time_place_manner = ["ddim", "fyth", "flwyddyn", "ddydd", "fis", "flynyddoedd", "fore", "drannoeth", "bryd", "droeon", "dramor", "ledled", "drwy", "gynt", "gynta", "gyntaf", "fin", "gwbl"]

def assign_specific_trigger(row) :
    """Function to identify specific mutation trigger.
    If lexical trigger: trigger is previous word.
    If morphosyntactic trigger: trigger can be direct object, vocative, adjective before noun, etc.
    Takes row and returns specific trigger."""

    word = row["correct_form"]
    lemma = row["lemma"]
    mut = row["mutation_type"]
    trig_type = row["trigger_type"]
    pos = row["pos"]
    dep_rel = row["dep_rel"]
    features = row.get("features") or {}
    verb_form = features.get("VerbForm")
    gender = features.get("Gender")
    number = features.get("Number")
    prev_word = row["prev_word"]
    prev_lemma = row["prev_lemma"]
    prev_pos = row.get("prev_pos")
    prev_feats = row.get("prev_feats") or {}
    prev_gender = prev_feats.get("Gender")
    prev_number = prev_feats.get("Number")
    before_previous_word = row["before_previous_word"]
    head_word = row["head_word"]
    head_feats = row["head_feats"] or {}
    head_gender = head_feats.get("Gender") or {}
    head_number = head_feats.get("Number") or {}

    # Lexical triggers
    if trig_type == "L" :
        if prev_word in {'"', '"', '"', '-'}: # dealing with cases of punctuation before mutated word
            return before_previous_word
        else :
            return prev_word

    # Morphosyntactic triggers of aspirate mutation (all negative zero trigger)
    if mut == "AM" and trig_type == "MS" :
        return "neg_zero_trigger_AM"

    # Morphosyntactic triggers of soft mutation
    if mut == "SM" and trig_type == "MS" :
    
        # Cases where preceding word is punctuation
        if prev_word in {'"', '"', '"', '-'} :
            prev_word = before_previous_word

        # yn as complement marker before noun or adjective but not place
        if prev_word and prev_word.lower() in {"yn", "'n"} : # and word is noun or adj but not place - "not place" already accounted for because NM not SM
            if pos in {"ADJ", "NOUN"} :
                return "yn_complement_adj_noun"
            
        # Common words that interrupt normal VSO word order
        if prev_word and prev_word.lower() in {"hefyd", "yma", "yna", "yno"} :
            return "non_normal_word_order"

        # Def article before two
        if word.lower() in {"ddau", "ddwy"} and prev_word and prev_word.lower() in {"y", "'r"} :
            return "two_def_article"

        # Def article before feminine singular noun
        if prev_word and prev_word.lower() in {"y", "'r", "yr"}: # and word is feminine and singular
            if gender == "Fem" and number == "Sing" :
                return "def_art_fem_sing_noun"
            else : # backup because some gender/number errors in Treebank. Using lexicon list from start of script
                if word.lower() in femsingnouns_lexicon :
                    return "def_art_fem_sing_noun"
                
        # Some ordinal and cardinal numbers before feminine singular noun
        if prev_word and prev_word.lower() in {"un", "pumed", "chweched"} :
            if gender == "Fem" and number == "Sing" :
                return "num_fem_sing_noun"
            else :
                if word.lower() in femsingnouns_lexicon :
                    return "num_fem_sing_noun"

        # Noun preceded by adjective
        if pos == "NOUN" and prev_pos == "ADJ" and verb_form != "Vnoun" :
            return "adj_before_noun"

        # Adjective or noun modifying feminine singular noun
        if (pos in {"ADJ", "NOUN"} or dep_rel in {"amod", "nmod"}) and verb_form != "Vnoun" :
            if head_gender == "Fem" and head_number == "Sing" :
                return "fem_sing_noun_modifier"
            elif prev_gender == "Fem" and prev_number == "Sing" : # Because some of the head annotations are wrong
                return "fem_sing_noun_modifier"
            else :
                if head_word and head_word.lower() in femsingnouns_lexicon :
                    return "fem_sing_noun_modifier"
                if prev_word and prev_word.lower() in femsingnouns_lexicon :
                    return "fem_sing_noun_modifier"

        # Vocative
        if dep_rel == "vocative" :
            return "vocative"
        
        # Inflected verb - zero trigger - omitted "fe", "mi", negative "ni"
        if pos == "VERB" and verb_form == "Fin" and dep_rel != "acl:relcl" : # main verbs only
            return "inflected_zero_trigger"

        # Direct object - includes "initial segment of whatever constituent is the structural complement to v"
        if dep_rel == "obj" or dep_rel == "ccomp" or dep_rel == "xcomp" or dep_rel == "cop" :
            return "direct_object_mutation"
        
        # Adverbs of time and place
        if word.lower() in adverbs_time_place_manner :
            return "adverb_time_place_manner"
        if word.lower() in {"ddechrau", "ddiwedd"} and pos == "NOUN" : # noun to target e.g. "start/end of February" and not verb "start/end"
            return "adverb_time_place_manner"
        
        # i + NP + ...
        if prev_word in {"iddo", "iddi", "iddyn", "iddynt"} : # iddo = i 'to' + fe 'him/it', iddi = i 'to' + hi 'her/it', etc.
            return "i_plus_NP"
        if prev_word in {"fi", "mi", "ti", "ef", "e", "hi", "fe", "fo", "ni", "chi", "chdi", "nhw"} and before_previous_word in {"i", "iddo", "iddi", "iddyn", "iddynt"} :
            return "i_plus_NP"
        if prev_pos in {"NOUN", "PROPN", "PRON"} and before_previous_word == "i" :
            return "i_plus_NP"
        
        # Relative clause - zero trigger (omitted "a")
        if dep_rel == "acl:relcl" :
            return "relative_zero_trigger"
   
        # Fallback
        return "UNKNOWN"

    return None

# Applying function to identify specific trigger type

df["specific_trigger"] = df.apply(assign_specific_trigger, axis=1)

# Printing counts of each specific trigger type within MS triggers - seeing how many left unknown

ms_df = df[df["trigger_type"] == "MS"]
trigger_counts = ms_df["specific_trigger"].value_counts()
print("Initial specific trigger counts:", trigger_counts, "\n-------")

"""
# Diagnosis of unknown triggers: exporting csv for manual annotation

df_manual = df[df["specific_trigger"] == "UNKNOWN"]
df_manual.to_csv("manual_annotations.csv", index=False)

"""

# Importing manual annotations 

url_manual_1 = "https://raw.githubusercontent.com/bethancunningham/tfm/main/manual_annotations/manual_annotations_final.csv"
manual_df_1 = pd.read_csv(url_manual_1, sep=";")
url_manual_2 = "https://raw.githubusercontent.com/bethancunningham/tfm/main/manual_annotations/manual_annotations2_final.csv"
manual_df_2 = pd.read_csv(url_manual_2, sep=";")

manual_df = pd.concat([manual_df_1, manual_df_2]).reset_index(drop=True)

# Building dictionary with manually entered specific triggers from csv

manual_lookup = {
    (row["sent_id"], row["token_id"]): row["specific_trigger"]
    for _, row in manual_df.iterrows()
    }

def apply_manual_fallback(row):
    """Function to find rows in df where specific_trigger is UNKNOWN and get manually entered specific_trigger in lookup dictionary.
    Returns manually entered specific_trigger"""
    if row["specific_trigger"] == "UNKNOWN":
        return manual_lookup.get((row["sent_id"], row["token_id"]), "UNKNOWN") # if not found still - returns UNKNOWN
    return row["specific_trigger"]

df["specific_trigger"] = df.apply(apply_manual_fallback, axis=1)

# Removing rows with lexicalised mutations and errors

df = df[~df["specific_trigger"].isin(["lexicalised", "lexicalised_draw", "lexicalised_remove", "error"])]

# Getting final specific_trigger counts after manual annotation

new_ms_df = df[df["trigger_type"] == "MS"]
new_trigger_counts = new_ms_df["specific_trigger"].value_counts()
print("After manual annotation:")
print("Specific trigger counts:\n", new_trigger_counts)
print("No. of sentences:\n", df["sent_id"].nunique())
print("No. instances of mutation:\n", len(df))
print("Mutation type counts:\n", df["mutation_type"].value_counts(), "\n-------")

# Exporting csv to annotate final unknown triggers

# df_manual = df[df["specific_trigger"] == "UNKNOWN"]
# df_manual.to_csv("manual_annotations_2.csv", index=False)


# Creating df where the correct version of the word is unmutated, and the incorrect one is mutated
# I will use a sample of the same lexical items seen in my SM, NM and AM examples, but unmutated, by searching for sentences with these unmutated words in the whole Treebank 

# Taking sample of 150 SM, NM and AM from df and creating lists of unmutated forms - taking 150 then 100 in case some words not in Treebank

unmutated = {}
for mut_type in ["SM", "NM", "AM"]:
    sample = df[df["mutation_type"] == mut_type].sample(n=150, random_state=123)
    unmutated[mut_type] = set(sample["incorrect_form"].tolist())  # set for faster lookup

# Searching for unmutated forms in Treebank and creating dfs with results

no_mutation_data = {"SM": [], "NM": [], "AM": []}

for sentence in merged_tokenlists :
    sent_id = sentence.metadata.get("sent_id")
    text = sentence.metadata.get("text")

    for i, token in enumerate(sentence) :
        prev_token = sentence[i-1] if i > 0 else None
        for mut_type, unmutated_list in unmutated.items():
            if token["form"] in unmutated_list:
                no_mutation_data[mut_type].append({
                    "sent_id": sent_id,
                    "sentence": text,
                    "sentence_tokens": [t["form"] for t in sentence],
                    "sentence_en": sentence.metadata.get("text_en"),
                    "correct_form": token["form"],
                    "lemma": token["lemma"],
                    "token_id": token["id"],
                    "pos": token["upos"],
                    "prev_word": sentence[i-1]["form"] if i > 0 else None,
                    "prev_lemma": sentence[i-1]["lemma"] if i > 0 else None,
                    "prev_pos": prev_token["upos"] if prev_token else None,
                })

# Sampling 100 from each and assigning mutation type label

no_mutation_samples = []
for mut_type, data in no_mutation_data.items() :
    sample = pd.DataFrame(data).sample(n=100, random_state=123)
    sample["mutation_type"] = f"no_{mut_type}"
    no_mutation_samples.append(sample)

# Creating df with no_mutation examples

df_no_mutation = pd.concat(no_mutation_samples).sample(frac=1, random_state=123).reset_index(drop=True)

# Checking there are 100 of each

print("Mutation type counts no_mutation sample:", df_no_mutation["mutation_type"].value_counts(), "\n-------")

# Mutating (correct) unmutated form to create incorrect forms

def apply_mutations(root, exclude=None):
    """Function to return {mut_type: mutated_form} for all applicable mutations of root word."""
    w = root.lower()
    cap = lambda s: s[0].upper() + s[1:] if root[0].isupper() else s
    key = next((k for k in ("ll", "rh", "c", "p", "t", "g", "b", "d", "m") if w.startswith(k)), None)
    if key is None :
        return {}
    else : # looking up in mutation rules dict
        return {
        mut: cap(prefix + w[strip:]) # strip removes first letter or two depending on dictionary key
        for mut, (prefix, strip) in MUTATION_RULES[key].items()
        if (prefix + w[strip:]) and mut != exclude # skips mutations already in df, e.g. if SM correct only provides NM and AM, if applicable
    }


# Adding extra column with incorrect word (mutated)

df_no_mutation["incorrect_form"] = df_no_mutation.apply(
    lambda row: apply_mutations(row["correct_form"]).get(row["mutation_type"].replace("no_", "")),
    axis=1
)

# Filling trigger_type and specific_trigger columns

df_no_mutation["trigger_type"] = "no_mutation"
df_no_mutation["specific_trigger"] = "no_mutation"

# Creating version of original df with mutated correct words ("df") just with necessary columns

df_mutation = df[["sent_id", "sentence", "sentence_tokens", "sentence_en", "correct_form", "lemma",
                   "mutation_type", "trigger_type", "specific_trigger", "token_id", "pos",
                   "prev_word", "prev_lemma", "prev_pos", "incorrect_form"]]

# Adding df_mutation and df_no_mutation together and shuffling to create final df

df_final = pd.concat([df_mutation, df_no_mutation]).sample(frac=1, random_state=123).reset_index(drop=True)

# Checking mutation type counts in final df

print("Mutation type counts with mutation and no_mutation items:", df_final["mutation_type"].value_counts(), "\n-------")

# Making incorrect sentences for minimal pairs

def make_incorrect_sentence(row) :
    """Function to replace correct_form in sentence with incorrect_form to create incorrect_sentence for minimal pair. Ensures that correct instance of correct_form
    is selected where the word is repeated in the sentence, based on its location in the sentence and its previous word."""
    sentence = row["sentence"]
    correct = row["correct_form"]
    incorrect = row["incorrect_form"]
    prev = row["prev_word"]
    token_id = row["token_id"]

    # Building list of (prev_context, word) patterns to try to ensure correct occurrence of word

    contexts = []
    if prev:
        contexts.append(prev)
        if prev in {"e", "hi", "nhw", "hwy"}:  # iddo/iddi/iddynt split in treebank - "iddo" is tokenised as "i fe", "iddi" as "i hi", etc.
            contexts.extend(["iddo", "iddi", "iddynt", "iddyn"])

    def find_matches(context):
        """Function to search for word plus its context to ensure the correct occurrence of the word. Returns matches.""" # Needed because a word can appear more than once in the sentence - need to find right one
        # Matching context followed by optional quote/number then the word
        pattern = re.escape(context) + r"\s[\u201c\u2018]?\d*" + re.escape(correct) + r"(?!\w)"  # Looking for context + space + optional quotes + optional digits + word + (negative lookup - not part of larger word)
        return [(m.start() + m.group().index(correct), m.start() + m.group().index(correct) + len(correct))
                for m in re.finditer(pattern, sentence)]

    all_matches = []
    for ctxt in contexts :
        all_matches = find_matches(ctxt)
        if all_matches:
            break

    # Fallback: just finding the word on its own if no matches
    if not all_matches :
        all_matches = [(m.start(), m.end()) for m in re.finditer(r"(?<!\w)" + re.escape(correct) + r"(?!\w)", sentence)]  # Finding start and end position of every occurrence of the word on its own

    if not all_matches :
        # print(f"Warning: 0 matches found for '{correct}' in: {sentence}")
        return sentence

    if len(all_matches) == 1 :
        start, end = all_matches[0]  # If only 1 match - take that match
    else:
        approx_pos = len(" ".join(sentence.split()[:token_id - 1]))
        start, end = min(all_matches, key=lambda m: abs(m[0] - approx_pos))  # If more than one match - find closest to the approx position of the word in the sequence of tokens - necessary because Treebank tokenises differently

    return sentence[:start] + incorrect + sentence[end:]  # Returns incorrect sentence

df_final["incorrect_sentence"] = df_final.apply(make_incorrect_sentence, axis=1)

# Checking there is only one different token between correct and incorrect sentences - should be 1

def check_one_diff(row):
    """Function to check number of different tokens between correct and incorrect sentences - should be 1"""
    orig   = row["sentence"].split()
    incorr = row["incorrect_sentence"].split()
    if len(orig) != len(incorr) :
        return "different_length"
    return sum(a != b for a, b in zip(orig, incorr))

df_final["num_diffs"] = df_final.apply(check_one_diff, axis=1)
# print(df_final["num_diffs"].value_counts())

# Removing instances where make_incorrect_sentence has failed - they are sentences with typos, cases of "mono/monon/monyn" (not really mutation) and "g" for century (not full word)

df_final = df_final[df_final["num_diffs"] == 1].reset_index(drop=True) # Only keeping rows where num_diffs is 1
print("Number of rows left after removal of errors:", len(df_final))
print("Mutation type counts after removal of errors:", df_final["mutation_type"].value_counts(), "\n-------")

print("Mutation type counts in initial df:\n", df_final["mutation_type"].value_counts())
print("Trigger type counts in initial df:\n", df_final["trigger_type"].value_counts(), "\n-------")

# Printing various counts for later error analysis
print("Specific trigger counts where trigger_type is MS:\n", df_final[df_final["trigger_type"] == "MS"]["specific_trigger"].value_counts())
print("Top 20 specific trigger counts where trigger_type is L:\n", df_final[df_final["trigger_type"] == "L"]["specific_trigger"].value_counts().head(20))
total = len(df_final)
print("Specific trigger counts where trigger_type is MS in pct:\n",
      (df_final[df_final["trigger_type"] == "MS"]["specific_trigger"].value_counts() / total * 100).round(2))
print("Top 20 specific trigger counts where trigger_type is L in pct:\n",
      (df_final[df_final["trigger_type"] == "L"]["specific_trigger"].value_counts().head(20) / total * 100).round(2))
cap_not_sentence_initial = df_final[
    (df_final["correct_form"].str[0].str.isupper()) &
    ((df_final["prev_word"].notna()) |  (df_final["pos"] == "PROPN"))
]
print("Capitalised (excl. non-PROPN sentence-initial):", len(cap_not_sentence_initial))
pct = len(cap_not_sentence_initial) / len(df_final) * 100
print(f"Capitalised (excl. non-PROPN sentence-initial): {pct:.2f}%")

for prefix in ["g", "rh", "ll"]:
    count = df_final[(df_final["mutation_type"] == "SM") & 
                     (df_final["lemma"].str.lower().str.startswith(prefix))]["lemma"].count()
    print(f"SM items with lemma starting with '{prefix}': {count}")
print("-------\n")
print("Rows in initial df:\n", len(df_final))
print("Exporting initial df to csv (name: initial_treebank_dataset.csv)")
df_final.to_csv("initial_treebank_dataset.csv", index=True)
print("-------\n")

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

df_expanded = pd.DataFrame(
    r for _, row in df_final.iterrows() for r in expand_rows(row)
).reset_index(drop=True)

df_expanded["num_diffs"] = df_expanded.apply(check_one_diff, axis=1)
# print(df_expanded["num_diffs"].value_counts()) # checking only 1 diff between correct and incorrect sentence in expanded df - yes

# Checking mutation type counts in expanded df

print("Mutation type counts in expanded df:\n", df_expanded["mutation_type"].value_counts())
print("Trigger type counts in expanded df:\n", df_expanded["trigger_type"].value_counts())
print("Incorrect form mutation types in expanded df:\n", df_expanded["incorrect_form_mut_type"].value_counts())
print("Rows in expanded df:\n", len(df_expanded))
print("Exporting final expanded df to csv (name: expanded_treebank_dataset.csv)")
df_expanded.to_csv("expanded_treebank_dataset.csv", index=True)
print("-------\n")

# Calculating average sentence length and type/token ratio for whole corpus (treebank) and sentences in my dataframe (with instances of mutation)

def tokenise_welsh(text): # simple tokeniser by space and apostrophe adequate for this purpose
    # First split on whitespace
    words = str(text).lower().split()
    tokens = []
    for word in words:
        # Splitting off clitics and contractions like 'r, 'n, 'm, 'th, 'ch, 'i
        parts = re.split(r"(?<=\w)('r|'n|'m|'th|'ch|'i|'w)\b", word)
        tokens.extend([p for p in parts if p])
    return tokens

df_final["sentence_tokens_tokeniser"] = df_final["sentence"].apply(tokenise_welsh)

df_final_tokens = [t for tokens in df_final.drop_duplicates("sent_id")["sentence_tokens_tokeniser"] for t in tokens]
n_final_sents   = df_final["sent_id"].nunique()

all_tokens = [t for sentence in merged_tokenlists for t in tokenise_welsh(sentence.metadata["text"])]

print("All sentences in Treebank")
print(f"Average sentence length: {len(all_tokens) / len(merged_tokenlists)}")
print(f"Type/token ratio: {len(set(all_tokens)) / len(all_tokens)}")
print(f"Herdan's C: {log(len(set(all_tokens))) / log(len(all_tokens))}\n-------\n")

print("\nMutation df sentences")
print(f"Average sentence length: {len(df_final_tokens) / n_final_sents}")
print(f"Type/token ratio: {len(set(df_final_tokens)) / len(df_final_tokens)}")
print(f"Herdan's C: {log(len(set(df_final_tokens))) / log(len(df_final_tokens))}\n-------\n")

# Sampling 500 questions from df_final (one row per question) for human test

human_no_NM = df_final[df_final['mutation_type'] == 'no_NM'].sample(n=67, random_state=11)
human_no_SM = df_final[df_final['mutation_type'] == 'no_SM'].sample(n=67, random_state=11)
human_no_AM = df_final[df_final['mutation_type'] == 'no_AM'].sample(n=66, random_state=11)
human_SM = df_final[df_final['mutation_type'] == 'SM'].sample(n=150, random_state=11)
human_NM = df_final[df_final['mutation_type'] == 'NM'].sample(n=75, random_state=11)
human_AM = df_final[df_final['mutation_type'] == 'AM'].sample(n=75, random_state=11)

all_500 = pd.concat([human_no_NM, human_no_SM, human_no_AM, human_SM, human_NM, human_AM])
all_500 = all_500.sample(frac=1, random_state=11).reset_index(drop=True)

# Splitting into two halves

half_1 = all_500.iloc[:250]
half_2 = all_500.iloc[250:]

# Pulling all rows from df_expanded matching each half to make multiple choice questions

def get_expanded(df_expanded, keys_df):
    keys = keys_df[['sent_id', 'token_id']].drop_duplicates()
    return df_expanded.merge(keys, on=['sent_id', 'token_id'], how='inner').reset_index(drop=True)

subject_1 = get_expanded(df_expanded, half_1).reset_index(drop=True)
subject_2 = get_expanded(df_expanded, half_2).reset_index(drop=True)

subject_1.to_csv('subject_1.csv', index=False)
subject_2.to_csv('subject_2.csv', index=False)

