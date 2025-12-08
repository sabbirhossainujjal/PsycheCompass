#!/usr/bin/env python3
"""
EDA script for CounselChat-like CSVs.

Produces:
 - eda_summary.csv (column summaries)
 - eda_text_stats.csv (per-row text features)
 - merged_with_text_stats.csv (original rows with appended text stats)
 - top_question_words.csv, top_answer_words.csv, top_question_bigrams.csv
 - therapist_counts.csv, topic_counts.csv, split_counts.csv
 - flagged_suicide_samples.csv
 - plots: question_len_hist.png, answer_len_hist.png, topic_counts.png, upvotes_views_scatter.png

Usage:
    python eda_counselchat.py --input data/counselchat.csv --outdir /path/to/outdir
    the preferred in our case is, once in the mentalagent folder on the terminal:
    python notebooks/eda_counselchat.py --input data/20200325_counsel_chat.csv --outdir notebooks/outputs/counselChat

If input omitted, script looks for data/counselchat.csv; if that file is missing it will error.
"""

import os
import re
import argparse
import json
import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional dependencies (spaCy, textstat). Script will continue if not installed.
try:
    import spacy
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
except Exception:
    TEXTSTAT_AVAILABLE = False

# -------------------------
# Configuration / constants
# -------------------------
DEFAULT_CSV = "data/counselchat.csv"
SUICIDE_PATTERNS = [
    r"\bi want to die\b", r"\bi don't want to be here\b", r"\bi should just go\b",
    r"\bkill myself\b", r"\btired of living\b", r"\bi can't go on\b", r"\bwish i was dead\b",
    r"\bshouldn'?t be here\b", r"\bi want to end my life\b", r"\bsuicide\b", r"\bworthless\b"
]
STOPWORDS = {
    "the","and","a","an","to","of","in","that","it","is","i","you","for","this","with","on","be","have","are","was","as","but","they","we","not","or","your","my","so","do","what","how","if","at"
}
PUNCT_RE = re.compile(r"[^\w'\s]")

# -------------------------
# Logging config
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("EDA")

# -------------------------
# Helpers
# -------------------------
def simple_tokenize(text: str) -> List[str]:
    if pd.isna(text):
        return []
    t = PUNCT_RE.sub(" ", str(text).lower())
    toks = [w for w in t.split() if w.strip()]
    return toks

def avg_sentence_count(text: str) -> int:
    if pd.isna(text) or not str(text).strip():
        return 0
    sents = re.split(r'[.!?]+', str(text))
    sents = [s.strip() for s in sents if s.strip()]
    return len(sents)

def estimate_syllables(word: str) -> int:
    # heuristic syllable counter (fast, works in aggregate)
    word = re.sub(r'[^a-z]', '', word.lower())
    if not word:
        return 0
    vowels = "aeiouy"
    prev = False
    count = 0
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev:
            count += 1
        prev = is_v
    if word.endswith("e") and count > 1:
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    return max(1, count)

def flesch_reading_ease(text: str) -> float:
    if pd.isna(text) or not str(text).strip():
        return np.nan
    tokens = simple_tokenize(text)
    n_words = len(tokens)
    n_sentences = avg_sentence_count(text) or 1
    syllables = sum(estimate_syllables(w) for w in tokens) or 1
    score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words) if n_words else np.nan
    return round(float(score), 2) if not np.isnan(score) else np.nan

def detect_patterns(text: str, patterns: List[str]) -> bool:
    if pd.isna(text):
        return False
    t = str(text).lower()
    for p in patterns:
        if re.search(p, t):
            return True
    return False

EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

def find_pii(text: str) -> dict:
    """
    Very simple PII detection: emails, phones, plus spaCy NER if available.
    Returns dict with counts and examples.
    """
    res = {"emails": [], "phones": [], "entities": []}
    if pd.isna(text):
        return res
    s = str(text)
    res["emails"] = EMAIL_RE.findall(s)
    res["phones"] = PHONE_RE.findall(s)
    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(s)
            for ent in doc.ents:
                # add only certain types
                if ent.label_ in {"PERSON", "GPE", "ORG", "LOC"}:
                    res["entities"].append({"text": ent.text, "label": ent.label_})
        except Exception as e:
            # spaCy model missing: ignore gracefully
            logger.debug("spaCy NER failed: %s", e)
    return res

def top_n_words(texts: List[str], n: int = 25, stopwords: set = STOPWORDS) -> List[Tuple[str,int]]:
    c = Counter()
    for t in texts:
        for w in simple_tokenize(t):
            if w in stopwords:
                continue
            c[w] += 1
    return c.most_common(n)

def top_bigrams(texts: List[str], n: int = 25) -> List[Tuple[Tuple[str,str],int]]:
    c = Counter()
    for t in texts:
        toks = [w for w in simple_tokenize(t) if w not in STOPWORDS]
        for i in range(len(toks)-1):
            c[(toks[i], toks[i+1])] += 1
    return c.most_common(n)

# -------------------------
# Main EDA function
# -------------------------
def run_eda(input_csv: str, outdir: str, sample_limit: int = None):
    os.makedirs(outdir, exist_ok=True)
    logger.info("Loading CSV: %s", input_csv)
    df = pd.read_csv(input_csv)
    if sample_limit:
        logger.info("Sampling first %d rows for speed", sample_limit)
        df = df.iloc[:sample_limit].copy()

    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    logger.info("Columns: %s", df.columns.tolist())

    # ensure numeric dtypes for common fields
    for col in ['upvotes', 'views', 'questionID']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Basic overview
    overview = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicates_exact": int(df.duplicated().sum())
    }
    with open(os.path.join(outdir, "eda_overview.json"), "w") as f:
        json.dump(overview, f, indent=2)
    logger.info("Saved overview to eda_overview.json")

    # Column summaries
    col_summaries = []
    for col in df.columns:
        s = df[col]
        cs = {"column": col, "dtype": str(s.dtype), "n_unique": int(s.nunique(dropna=True)), "n_missing": int(s.isna().sum())}
        if pd.api.types.is_numeric_dtype(s):
            cs.update({
                "min": None if s.dropna().empty else float(s.min()),
                "max": None if s.dropna().empty else float(s.max()),
                "mean": None if s.dropna().empty else float(s.mean()),
                "median": None if s.dropna().empty else float(s.median()),
                "std": None if s.dropna().empty else float(s.std()),
            })
        else:
            vc = s.fillna("<<MISSING>>").value_counts()
            cs["top_values"] = vc.head(5).to_dict()
        col_summaries.append(cs)
    pd.DataFrame(col_summaries).to_csv(os.path.join(outdir, "eda_summary.csv"), index=False)
    logger.info("Saved column summaries to eda_summary.csv")

    # Text features (questionText and answerText + questionTitle)
    text_cols = [c for c in ["questionTitle","questionText","answerText"] if c in df.columns]
    text_stats_rows = []
    for idx, row in df.iterrows():
        qtext = row.get("questionText", "")
        atxt = row.get("answerText", "")
        q_tokens = simple_tokenize(qtext)
        a_tokens = simple_tokenize(atxt)
        q_wc = len(q_tokens)
        a_wc = len(a_tokens)
        q_sent = avg_sentence_count(qtext)
        a_sent = avg_sentence_count(atxt)
        q_read = None
        a_read = None
        if TEXTSTAT_AVAILABLE:
            try:
                q_read = textstat.flesch_reading_ease(qtext) if qtext and qtext.strip() else np.nan
                a_read = textstat.flesch_reading_ease(atxt) if atxt and atxt.strip() else np.nan
            except Exception:
                q_read = flesch_reading_ease(qtext)
                a_read = flesch_reading_ease(atxt)
        else:
            q_read = flesch_reading_ease(qtext)
            a_read = flesch_reading_ease(atxt)

        pii_q = find_pii(qtext)
        pii_a = find_pii(atxt)

        text_stats_rows.append({
            "index": int(idx),
            "question_word_count": int(q_wc),
            "answer_word_count": int(a_wc),
            "question_sentence_count": int(q_sent),
            "answer_sentence_count": int(a_sent),
            "question_readability": q_read,
            "answer_readability": a_read,
            "flagged_suicide_in_question": bool(detect_patterns(qtext, SUICIDE_PATTERNS)),
            "flagged_suicide_in_answer": bool(detect_patterns(atxt, SUICIDE_PATTERNS)),
            "question_emails_found": len(pii_q["emails"]),
            "question_phones_found": len(pii_q["phones"]),
            "answer_emails_found": len(pii_a["emails"]),
            "answer_phones_found": len(pii_a["phones"]),
            "question_entities_sample": json.dumps(pii_q.get("entities", [])[:3]),
            "answer_entities_sample": json.dumps(pii_a.get("entities", [])[:3]),
        })

    text_stats_df = pd.DataFrame(text_stats_rows)
    text_stats_df.to_csv(os.path.join(outdir, "eda_text_stats.csv"), index=False)
    logger.info("Saved per-row text stats to eda_text_stats.csv")

    # Merge for inspection
    merged = df.reset_index().merge(text_stats_df, left_on="index", right_on="index", how="left")
    merged.to_csv(os.path.join(outdir, "merged_with_text_stats.csv"), index=False)
    logger.info("Saved merged file to merged_with_text_stats.csv")

    # Topic / split / therapist analysis
    if 'topic' in df.columns:
        topic_counts = df['topic'].fillna("<<MISSING>>").value_counts().rename_axis('topic').reset_index(name='count')
        topic_counts.to_csv(os.path.join(outdir, "topic_counts.csv"), index=False)
    else:
        topic_counts = pd.DataFrame(columns=["topic","count"])
    if 'split' in df.columns:
        split_counts = df['split'].fillna("<<MISSING>>").value_counts().rename_axis('split').reset_index(name='count')
        split_counts.to_csv(os.path.join(outdir, "split_counts.csv"), index=False)
    else:
        split_counts = pd.DataFrame(columns=["split","count"])

    if 'therapistInfo' in df.columns:
        therapist_counts = df['therapistInfo'].fillna("<<MISSING>>").value_counts().reset_index().rename(columns={'index':'therapistInfo','therapistInfo':'count'})
        therapist_counts.to_csv(os.path.join(outdir, "therapist_counts.csv"), index=False)
    else:
        therapist_counts = pd.DataFrame(columns=["therapistInfo","count"])

    # Top words and bigrams
    top_questions = top_n_words(df['questionText'].fillna("").tolist(), n=50)
    top_answers = top_n_words(df['answerText'].fillna("").tolist(), n=80)
    pd.DataFrame(top_questions, columns=['word','count']).to_csv(os.path.join(outdir, "top_question_words.csv"), index=False)
    pd.DataFrame(top_answers, columns=['word','count']).to_csv(os.path.join(outdir, "top_answer_words.csv"), index=False)

    top_q_bigrams = top_bigrams(df['questionText'].fillna("").tolist(), n=50)
    top_q_bigrams_df = pd.DataFrame([{"bigram":" ".join(b),"count":c} for b,c in top_q_bigrams])
    top_q_bigrams_df.to_csv(os.path.join(outdir, "top_question_bigrams.csv"), index=False)

    # Upvotes / views analysis
    if 'upvotes' in df.columns and 'views' in df.columns:
        uv = df[['upvotes','views']].copy()
        uv['upvotes'] = pd.to_numeric(uv['upvotes'], errors='coerce').fillna(0)
        uv['views'] = pd.to_numeric(uv['views'], errors='coerce').fillna(0)
        uv['upvote_view_ratio'] = uv.apply(lambda r: (r['upvotes'] / r['views']) if r['views']>0 else 0, axis=1)
        uv.describe().to_csv(os.path.join(outdir, "upvotes_views_stats.csv"))
    else:
        uv = pd.DataFrame(columns=["upvotes","views"])

    # Flagged suicide rows
    flagged = merged[(merged['flagged_suicide_in_question']==True) | (merged['flagged_suicide_in_answer']==True)].copy()
    flagged.to_csv(os.path.join(outdir, "flagged_suicide_samples.csv"), index=False)
    logger.info("Saved flagged suicide samples (heuristic) to flagged_suicide_samples.csv")

    # Duplicates
    dup_exact = int(df.duplicated(subset=['questionTitle','questionText','answerText']).sum()) if set(['questionTitle','questionText','answerText']).issubset(df.columns) else int(df.duplicated().sum())
    pd.DataFrame([{"duplicate_exact_count": dup_exact}]).to_csv(os.path.join(outdir, "duplicates_summary.csv"), index=False)

    # Numeric correlations
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        corr = merged[numeric_cols].corr().round(3)
        corr.to_csv(os.path.join(outdir, "numeric_correlations.csv"))
    else:
        corr = pd.DataFrame()

    # -------------------------
    # Plots (matplotlib only)
    # -------------------------
    logger.info("Creating plots (matplotlib)...")
    # question length histogram
    if 'question_word_count' in text_stats_df.columns:
        plt.figure(figsize=(7,4))
        plt.hist(text_stats_df['question_word_count'].fillna(0), bins=20)
        plt.title("Question word count distribution")
        plt.xlabel("Words in question")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "question_len_hist.png"))
        plt.close()

    # answer length histogram
    if 'answer_word_count' in text_stats_df.columns:
        plt.figure(figsize=(7,4))
        plt.hist(text_stats_df['answer_word_count'].fillna(0), bins=20)
        plt.title("Answer word count distribution")
        plt.xlabel("Words in answer")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "answer_len_hist.png"))
        plt.close()

    # topic counts bar chart
    if not topic_counts.empty:
        plt.figure(figsize=(8,4))
        topics = topic_counts['topic'].tolist()
        counts = topic_counts['count'].tolist()
        plt.bar(range(len(topics)), counts)
        plt.xticks(range(len(topics)), topics, rotation=45, ha='right')
        plt.title("Topic counts")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "topic_counts.png"))
        plt.close()

    # upvotes vs views scatter
    if not uv.empty:
        plt.figure(figsize=(6,4))
        plt.scatter(uv['views'], uv['upvotes'])
        plt.xlabel("Views")
        plt.ylabel("Upvotes")
        plt.title("Upvotes vs Views")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "upvotes_views_scatter.png"))
        plt.close()

    # All outputs list
    created = sorted([os.path.join(outdir,f) for f in os.listdir(outdir)])
    logger.info("EDA complete. Files created in %s:", outdir)
    for f in created:
        logger.info(" - %s", f)

# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Detailed EDA for CounselChat-style CSV")
    p.add_argument("--input", "-i", type=str, default=DEFAULT_CSV, help="Path to CSV file")
    p.add_argument("--outdir", "-o", type=str, default="outputs/eda", help="Output directory for CSVs and plots")
    p.add_argument("--sample", "-s", type=int, default=None, help="Optional sample limit for quick runs")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        logger.error("Input CSV not found: %s", args.input)
        raise SystemExit(1)
    run_eda(args.input, args.outdir, sample_limit=args.sample)

if __name__ == "__main__":
    main()
