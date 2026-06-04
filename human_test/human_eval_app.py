"""
Welsh Mutation Evaluation — Streamlit app.

Subjects pick 1 or 2 at the start, then answer multiple-choice questions.
Results are written to Google Sheets after every answer.

Deploy to Streamlit Cloud and share the URL.
"""

import re
import random
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Welsh Grammaticality Evaluation", page_icon="🏴󠁧󠁢󠁷󠁬󠁳󠁿")

# ── Google Sheets setup ───────────────────────────────────────────────────────

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

@st.cache_resource
def get_sheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"], scopes=SCOPES
    )
    client = gspread.authorize(creds)
    return client.open_by_key(st.secrets["spreadsheet_id"]).sheet1

def write_response(sheet, subject: int, question_num: int, sent_id: str,
                   token_id, correct_form: str, human_choice: str,
                   chose_correct: int, mutation_type: str):
    sheet.append_row([
        datetime.utcnow().isoformat(timespec="seconds"),
        subject,
        question_num,
        str(sent_id),
        int(token_id),
        correct_form,
        human_choice,
        chose_correct,
        mutation_type,
    ])

# ── Data loading ──────────────────────────────────────────────────────────────

DATA_URLS = {
    1: st.secrets["subject_1_url"],
    2: st.secrets["subject_2_url"],
}

@st.cache_data
def load_questions(subject: int) -> list[dict]:
    df = pd.read_csv(DATA_URLS[subject])
    return build_questions(df)

# ── Question builder (same logic as terminal script) ─────────────────────────

def make_gapped(sentence: str, correct_form: str, prev_word,
                token_id: int) -> str:
    contexts = [prev_word] if prev_word and pd.notna(prev_word) else []
    if prev_word in {"e", "hi", "nhw", "hwy"}:
        contexts += ["iddo", "iddi", "iddynt", "iddyn"]

    def find_matches(context):
        pattern = (re.escape(str(context)) + r"\s[\u201c\u2018]?\d*"
                   + re.escape(correct_form) + r"(?!\w)")
        return [(m.start() + m.group().index(correct_form),
                 m.start() + m.group().index(correct_form) + len(correct_form))
                for m in re.finditer(pattern, sentence)]

    all_matches = []
    for ctx in contexts:
        all_matches = find_matches(ctx)
        if all_matches:
            break

    if not all_matches:
        all_matches = [(m.start(), m.end()) for m in re.finditer(
            r"(?<!\w)" + re.escape(correct_form) + r"(?!\w)", sentence)]

    if not all_matches:
        return sentence.replace(correct_form, "___", 1)

    if len(all_matches) == 1:
        start, end = all_matches[0]
    else:
        approx_pos = len(" ".join(sentence.split()[:int(token_id) - 1]))
        start, end = min(all_matches, key=lambda m: abs(m[0] - approx_pos))

    return sentence[:start] + "______" + sentence[end:]


def build_questions(df: pd.DataFrame) -> list[dict]:
    questions = []
    for (sent_id, token_id), group in df.groupby(["sent_id", "token_id"], sort=False):
        first = group.iloc[0]
        correct_form  = first["correct_form"]
        sentence      = first["sentence"]
        prev_word     = first.get("prev_word")
        mutation_type = first["mutation_type"]

        gapped = make_gapped(sentence, correct_form, prev_word, int(token_id))

        incorrect_options = []
        for _, row in group.iterrows():
            inc = row["incorrect_form"]
            if pd.notna(inc) and inc != correct_form:
                incorrect_options.append(inc)

        if not incorrect_options:
            continue

        all_options = [correct_form] + list(dict.fromkeys(incorrect_options))
        random.shuffle(all_options)
        correct_key = all_options.index(correct_form)

        questions.append({
            "sent_id":       sent_id,
            "token_id":      token_id,
            "gapped":        gapped,
            "correct_form":  correct_form,
            "options":       all_options,
            "correct_key":   correct_key,
            "mutation_type": mutation_type,
        })
    return questions

# ── Session state helpers ─────────────────────────────────────────────────────

def init_state():
    defaults = {
        "subject":       None,
        "questions":     [],
        "current":       0,
        "score":         0,
        "done":          False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    init_state()

    st.title("Welsh Grammaticality Evaluation 🏴󠁧󠁢󠁷󠁬󠁳󠁿")

    # ── Subject selection ─────────────────────────────────────────────────────
    if st.session_state.subject is None:
        st.markdown("Please select your participant number to begin.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Participant 1", use_container_width=True, type="primary"):
                st.session_state.subject = 1
                st.session_state.questions = load_questions(1)
                st.rerun()
        with col2:
            if st.button("Participant 2", use_container_width=True, type="primary"):
                st.session_state.subject = 2
                st.session_state.questions = load_questions(2)
                st.rerun()
        return

    # ── Finished screen ───────────────────────────────────────────────────────
    if st.session_state.done:
        total = len(st.session_state.questions)
        st.success(f"All done! You answered all {total} questions.")
        st.markdown("You can now close this tab. Thank you for participating!")
        return

    # ── Question screen ───────────────────────────────────────────────────────
    questions = st.session_state.questions
    current   = st.session_state.current

    if current >= len(questions):
        st.session_state.done = True
        st.rerun()

    q = questions[current]
    total = len(questions)

    # Progress bar
    st.progress(current / total, text=f"Question {current + 1} of {total}")
    st.markdown("---")

    # Instruction
    st.markdown("**Which word correctly fills the gap?**")
    st.markdown("")

    # Sentence with gap — display in a styled box
    st.markdown(
        f"""<div style="
            background: var(--background-color);
            border: 1px solid rgba(128,128,128,0.3);
            border-radius: 8px;
            padding: 16px 20px;
            font-size: 17px;
            line-height: 1.7;
            margin-bottom: 20px;
        ">{q['gapped']}</div>""",
        unsafe_allow_html=True,
    )

    # Answer buttons — one per option
    sheet = get_sheet()
    cols = st.columns(len(q["options"]))
    for i, (col, option) in enumerate(zip(cols, q["options"])):
        with col:
            if st.button(option, key=f"opt_{current}_{i}", use_container_width=True):
                chose_correct = int(i == q["correct_key"])
                st.session_state.score += chose_correct

                # Write to Google Sheets
                write_response(
                    sheet,
                    subject       = st.session_state.subject,
                    question_num  = current + 1,
                    sent_id       = q["sent_id"],
                    token_id      = q["token_id"],
                    correct_form  = q["correct_form"],
                    human_choice  = option,
                    chose_correct = chose_correct,
                    mutation_type = q["mutation_type"],
                )

                st.session_state.current += 1
                st.rerun()

    st.markdown("---")
    st.caption(f"Participant {st.session_state.subject}")


if __name__ == "__main__":
    main()
