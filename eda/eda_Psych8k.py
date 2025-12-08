#!/usr/bin/env python3
"""
EDA for EmoCareAI/Psych8k (Hugging Face dataset).

Produces:
 - eda_overview.json
 - eda_summary.csv
 - eda_text_stats.csv
 - duplicates_summary.csv
 - top_instruction_words.csv
 - top_input_words.csv
 - top_output_words.csv
 - top_instruction_bigrams.csv
 - flagged_suicide_samples.csv
 - plots: instruction_len_hist.png, input_len_hist.png, output_len_hist.png, topic_counts.png

Usage:
  # Windows PowerShell (temporary session)
  $env:HUGGINGFACE_API_KEY = "hf_xxx"
  python eda_psych8k.py --outdir outputs/psych8k --use_auth_token

If you prefer .env, use python-dotenv and set HF token there.
"""
import os
import re
import json
import argparse
import logging
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download("punkt", quiet=True)
from datasets import load_dataset

# ---------- config ------------
SUICIDE_PATTERNS = [
    r"\bi want to die\b", r"\bkill myself\b", r"\bwant to end my life\b",
    r"\bi wish i was dead\b", r"\bi can't go on\b", r"\bsuicide\b",
    r"\bshouldn't be here\b", r"\bi don't want to be here\b", r"\bi want to die\b"
]
PII_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PII_PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,4}")

STOPWORDS = set([
    "the","and","a","an","to","of","in","that","it","is","i","you","for","this","with","on",
    "be","have","are","was","as","but","they","we","not","or","your","my","so","do","what","how",
    "if","at","me","im","ive","dont","can't"
])

PUNCT_RE = re.compile(r"[^\w'\s]")

logger = logging.getLogger("eda_psych8k")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------- helpers ------------
def simple_tokenize(text: str) -> List[str]:
    if text is None:
        return []
    t = PUNCT_RE.sub(" ", text.lower())
    toks = [w for w in t.split() if w.strip()]
    return toks

def top_n_words(texts: List[str], n: int = 50, stopwords: set = STOPWORDS) -> List[Tuple[str,int]]:
    c = Counter()
    for t in texts:
        for w in simple_tokenize(t):
            if w in stopwords: continue
            c[w] += 1
    return c.most_common(n)

def top_bigrams(texts: List[str], n: int = 50, stopwords: set = STOPWORDS) -> List[Tuple[Tuple[str,str],int]]:
    c = Counter()
    for t in texts:
        toks = [w for w in simple_tokenize(t) if w not in stopwords]
        for i in range(len(toks)-1):
            c[(toks[i], toks[i+1])] += 1
    return c.most_common(n)

def detect_patterns(text: str, patterns: List[str]) -> bool:
    if not isinstance(text, str):
        return False
    s = text.lower()
    for p in patterns:
        if re.search(p, s):
            return True
    return False

def find_pii(text: str) -> Dict[str, List[str]]:
    res = {"emails": [], "phones": []}
    if not isinstance(text, str):
        return res
    res["emails"] = PII_EMAIL_RE.findall(text)
    res["phones"] = PII_PHONE_RE.findall(text)
    return res

# ---------- main ----------
def run_eda(outdir: str, use_auth_token: bool = False):
    os.makedirs(outdir, exist_ok=True)

    # Load dataset (note: may require auth token)
    logger.info("Loading Hugging Face dataset: EmoCareAI/Psych8k")
    ds = load_dataset("EmoCareAI/Psych8k", token=True)# , use_auth_token=use_auth_token)
    # dataset may have splits; merge into a dataframe for EDA
    # find first split name
    split_names = list(ds.keys())
    logger.info("Available splits: %s", split_names)
    full = []
    for sp in split_names:
        pdf = pd.DataFrame(ds[sp])
        pdf["__split__"] = sp
        full.append(pdf)
    df = pd.concat(full, ignore_index=True, sort=False)
    logger.info("Loaded rows: %d", len(df))

    # Columns expected: instruction, input, output
    cols_present = df.columns.tolist()
    logger.info("Columns present: %s", cols_present)

    # Overview
    overview = {
        "n_rows": int(len(df)),
        "n_cols": int(len(cols_present)),
        "columns": cols_present,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "sample_rows": df.head(3).to_dict(orient="records")
    }
    with open(os.path.join(outdir, "eda_overview.json"), "w", encoding="utf8") as f:
        json.dump(overview, f, indent=2, ensure_ascii=False)

    # Basic column summaries
    col_summ = []
    for col in cols_present:
        s = df[col]
        cs = {"column": col, "dtype": str(s.dtype), "n_missing": int(s.isna().sum()), "n_unique": int(s.nunique(dropna=True))}
        if pd.api.types.is_string_dtype(s):
            vc = s.fillna("<<MISSING>>").value_counts().head(10).to_dict()
            cs["top_values"] = vc
        col_summ.append(cs)
    pd.DataFrame(col_summ).to_csv(os.path.join(outdir, "eda_summary.csv"), index=False)

    # Text stats for each row
    rows = []
    for idx, row in df.iterrows():
        instr = row.get("instruction","") or ""
        inp = row.get("input","") or ""
        outp = row.get("output","") or ""
        instr_tok = simple_tokenize(instr)
        inp_tok = simple_tokenize(inp)
        out_tok = simple_tokenize(outp)
        rows.append({
            "index": int(idx),
            "instruction_len_words": len(instr_tok),
            "input_len_words": len(inp_tok),
            "output_len_words": len(out_tok),
            "instruction_sents": len(nltk.tokenize.sent_tokenize(instr)),
            "input_sents": len(nltk.tokenize.sent_tokenize(inp)),
            "output_sents": len(nltk.tokenize.sent_tokenize(outp)),
            "instruction_readability": None,  # optional: add textstat if available
            "flagged_suicide_in_instruction": detect_patterns(instr, SUICIDE_PATTERNS),
            "flagged_suicide_in_input": detect_patterns(inp, SUICIDE_PATTERNS),
            "flagged_suicide_in_output": detect_patterns(outp, SUICIDE_PATTERNS),
            "instr_emails": find_pii(instr)["emails"],
            "instr_phones": find_pii(instr)["phones"],
            "input_emails": find_pii(inp)["emails"],
            "input_phones": find_pii(inp)["phones"],
            "output_emails": find_pii(outp)["emails"],
            "output_phones": find_pii(outp)["phones"],
            "__split__": row.get("__split__","")
        })
    text_stats = pd.DataFrame(rows)
    text_stats.to_csv(os.path.join(outdir, "eda_text_stats.csv"), index=False)

    # Merge summary
    merged = pd.concat([df.reset_index(drop=True), text_stats], axis=1)
    merged.to_csv(os.path.join(outdir, "merged_with_text_stats.csv"), index=False)

    # Duplicates
    dup_exact = int(df.duplicated(subset=["instruction","input","output"]).sum())
    pd.DataFrame([{"duplicate_exact_count": dup_exact}]).to_csv(os.path.join(outdir, "duplicates_summary.csv"), index=False)

    # Top words/bigrams
    pd.DataFrame(top_n_words(df["instruction"].fillna("").tolist(), n=200), columns=["word","count"]).to_csv(os.path.join(outdir, "top_instruction_words.csv"), index=False)
    pd.DataFrame(top_n_words(df["input"].fillna("").tolist(), n=200), columns=["word","count"]).to_csv(os.path.join(outdir, "top_input_words.csv"), index=False)
    pd.DataFrame(top_n_words(df["output"].fillna("").tolist(), n=400), columns=["word","count"]).to_csv(os.path.join(outdir, "top_output_words.csv"), index=False)

    tb = top_bigrams(df["instruction"].fillna("").tolist(), n=100)
    pd.DataFrame([{"bigram": " ".join(b), "count": c} for b,c in tb]).to_csv(os.path.join(outdir, "top_instruction_bigrams.csv"), index=False)

    # Flagged suicide samples
    flagged = merged[(merged["flagged_suicide_in_instruction"]==True) | (merged["flagged_suicide_in_input"]==True) | (merged["flagged_suicide_in_output"]==True)].copy()
    flagged.to_csv(os.path.join(outdir, "flagged_suicide_samples.csv"), index=False)

    # Basic plots
    plt.figure(figsize=(7,4))
    plt.hist(text_stats["instruction_len_words"].fillna(0), bins=30)
    plt.title("Instruction word count distribution")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "instruction_len_hist.png"))
    plt.close()

    plt.figure(figsize=(7,4))
    plt.hist(text_stats["input_len_words"].fillna(0), bins=30)
    plt.title("Input word count distribution")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "input_len_hist.png"))
    plt.close()

    plt.figure(figsize=(7,4))
    plt.hist(text_stats["output_len_words"].fillna(0), bins=30)
    plt.title("Output word count distribution")
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "output_len_hist.png"))
    plt.close()

    # Save a small sample for manual inspection by clinicians
    sample_n = min(200, len(merged))
    merged.sample(sample_n, random_state=42).to_csv(os.path.join(outdir, "random_sample_for_clinician_review.csv"), index=False)

    # numeric correlations
    numcols = ["instruction_len_words","input_len_words","output_len_words","instruction_sents","input_sents","output_sents"]
    corr = merged[numcols].corr().round(3)
    corr.to_csv(os.path.join(outdir, "numeric_correlations.csv"))

    logger.info("EDA complete. Files saved to %s", outdir)
    return outdir

# ---------- CLI ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="outputs/psych8k")
    p.add_argument("--use_auth_token", action="store_true", help="Add use_auth_token=True to datasets.load_dataset (for HF private datasets or gated access)")
    args = p.parse_args()
    run_eda(args.outdir, use_auth_token=args.use_auth_token)
