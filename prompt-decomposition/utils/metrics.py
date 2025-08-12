import re
import json
import textwrap
from difflib import SequenceMatcher
from sklearn.metrics import f1_score
import sys
import os
import pandas as pd
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def extract_number(text):
    match = re.search(r"\d+(\.\d+)?", text)
    return float(match.group()) if match else None

def evaluate_f1(gold, predicted):
    gold_set = set(normalize(q) for q in gold)
    pred_set = set(normalize(q) for q in predicted)

    true_positives = len(gold_set & pred_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gold_set) if gold_set else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def semantic_match(gold, predicted, threshold=0.75):
    if not gold or not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    gold_emb = model.encode(gold, convert_to_tensor=True).to(device)
    pred_emb = model.encode(predicted, convert_to_tensor=True).to(device)

    matched = 0
    used = set()
    for i, ge in enumerate(gold_emb):
        scores = util.cos_sim(ge, pred_emb)[0]
        for j, score in enumerate(scores):
            if j in used:
                continue
            if score >= threshold:
                matched += 1
                used.add(j)
                break

    precision = matched / len(predicted)
    recall = matched / len(gold)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def extract_subquestions(text):
    lines = text.split('\n')
    subq_lines = []
    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("subq") or stripped.endswith("?"):
            subq_lines.append(stripped)
    return subq_lines


def extract_answers(text):
    lines = text.split('\n')
    subans = []
    for line in lines:
        line = line.strip()
        if re.match(r"^a\d+[:\s]", line, re.IGNORECASE):
            subans.append(line)
    return subans


def extract_final_answer(text):
    lines = text.strip().split('\n')
    final_line = None

    for line in lines:
        if re.search(r'\bfinal answer\b', line, re.IGNORECASE):
            final_line = line
            break

    if not final_line:
        for line in reversed(lines):
            if re.search(r'\d', line) and not line.strip().lower().startswith("subq"):
                final_line = line
                break

    return final_line.strip() if final_line else None


def analyze_file(filepath):
    with open(filepath) as f:
        data = json.load(f)

    all_results = []
    for item in data:
        pred = item.get("3_shot_decomposition", "")
        gold = item.get("gold_decomposition", "")
        qid = item.get("id")

        # Extract components
        gold_subqs = extract_subquestions(gold)
        pred_subqs = extract_subquestions(pred)

        gold_subans = extract_answers(gold)
        pred_subans = extract_answers(pred)

        gold_final = extract_final_answer(gold)
        pred_final = extract_final_answer(pred)

        # Compute F1 and semantic metrics
        subq_metrics = evaluate_f1(gold_subqs, pred_subqs)
        subq_semantic = semantic_match(gold_subqs, pred_subqs)

        subans_metrics = evaluate_f1(gold_subans, pred_subans)
        subans_semantic = semantic_match(gold_subans, pred_subans)

        # Final answer metrics as singleton sets
        final_gold_list = [gold_final] if gold_final else []
        final_pred_list = [pred_final] if pred_final else []

        final_metrics = evaluate_f1(final_gold_list, final_pred_list)
        final_semantic = semantic_match(final_gold_list, final_pred_list)

        # Exact match and numeric match
        final_exact = int(normalize(gold_final or "") == normalize(pred_final or ""))
        final_numeric = int(extract_number(gold_final or "") == extract_number(pred_final or ""))

        # Exact match for full decomposition & subanswer comparison
        subq_exact_match = int(pred.strip() == gold.strip())
        subans_exact_match = int(" ".join(pred_subans).strip() == " ".join(gold_subans).strip())

        all_results.append({
            "id": qid,
            "subq_f1": subq_metrics["f1"],
            "subq_precision": subq_metrics["precision"],
            "subq_recall": subq_metrics["recall"],
            "subq_sem_f1": subq_semantic["f1"],
            "subq_sem_precision": subq_semantic["precision"],
            "subq_sem_recall": subq_semantic["recall"],
            "subq_exact_match": subq_exact_match,

            "subans_f1": subans_metrics["f1"],
            "subans_precision": subans_metrics["precision"],
            "subans_recall": subans_metrics["recall"],
            "subans_sem_f1": subans_semantic["f1"],
            "subans_sem_precision": subans_semantic["precision"],
            "subans_sem_recall": subans_semantic["recall"],
            "subans_exact_match": subans_exact_match,

            "final_f1": final_metrics["f1"],
            "final_precision": final_metrics["precision"],
            "final_recall": final_metrics["recall"],
            "final_sem_f1": final_semantic["f1"],
            "final_sem_precision": final_semantic["precision"],
            "final_sem_recall": final_semantic["recall"],
            "final_exact_match": final_exact,
            "final_numeric_match": final_numeric
        })

    return all_results



def summarize_metrics(static_metrics, random_metrics, dynamic_metrics):
    # Convert each list of dicts into DataFrames and label by few-shot type
    all_results = []
    for metrics, label in zip([static_metrics, random_metrics, dynamic_metrics],
                              ["static", "random", "dynamic"]):
        df = pd.DataFrame(metrics)
        df["few_shot_type"] = label
        all_results.append(df)

    # Concatenate all data
    full_df = pd.concat(all_results, ignore_index=True)

    # Define correct metric column names matching the analyze_file keys
    subquestion_columns = [
        "subq_exact_match",
        "subq_f1",
        "subq_precision",
        "subq_recall",
        "subq_sem_f1",
        "subq_sem_precision",
        "subq_sem_recall"
    ]

    subanswer_columns = [
        "subans_exact_match",
        "subans_f1",
        "subans_precision",
        "subans_recall",
        "subans_sem_f1",
        "subans_sem_precision",
        "subans_sem_recall"
    ]

    final_answer_columns = [
        "final_exact_match",
        "final_numeric_match",
        "final_f1",
        "final_precision",
        "final_recall",
        "final_sem_f1",
        "final_sem_precision",
        "final_sem_recall"
    ]

    # Filter columns based on availability
    subquestion_columns = [col for col in subquestion_columns if col in full_df.columns]
    subanswer_columns = [col for col in subanswer_columns if col in full_df.columns]
    final_answer_columns = [col for col in final_answer_columns if col in full_df.columns]

    # Group by few_shot_type and compute averages
    subq_summary = full_df.groupby("few_shot_type")[subquestion_columns].mean().round(3)
    subans_summary = full_df.groupby("few_shot_type")[subanswer_columns].mean().round(3)
    final_ans_summary = full_df.groupby("few_shot_type")[final_answer_columns].mean().round(3)

    return subq_summary, subans_summary, final_ans_summary


