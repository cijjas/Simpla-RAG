import json
import os
import csv
import sys
from pathlib import Path
from rag_metrics import RAGMetrics

# === Parse command-line argument ===
if len(sys.argv) != 2:
    print("Usage: python evaluate_custom.py <folder_number>")
    sys.exit(1)

folder_prefix = sys.argv[1]

# === Locate directory that starts with given prefix ===
base_dir = Path(".")
matching_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(folder_prefix)]

if not matching_dirs:
    print(f"❌ No directory found starting with '{folder_prefix}'")
    sys.exit(1)

target_dir = matching_dirs[0]  # take the first match

# === File paths ===
easy_path = target_dir / "ragas_easy.json"
hard_path = target_dir / "ragas_hard.json"
output_csv = Path("../results_custom.csv")

# === Initialize metric engine ===
metrics = RAGMetrics()

# === Evaluation logic ===
def evaluate_file(json_path, difficulty_tag):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        question = item["question"]
        contexts = item["contexts"]
        answer = item["answer"]
        reference = item["reference"]

        row_metrics = {
            "folder": target_dir.name,
            "difficulty": difficulty_tag,
            "question": question,
            "context_precision": metrics.context_precision(contexts, reference),
            "context_recall": metrics.context_recall(contexts, reference),
            "faithfulness": metrics.faithfulness(contexts, answer),
            "answer_relevance": metrics.answer_relevance(question, answer),
        }
        results.append(row_metrics)
    return results

# === Evaluate both ===
results_easy = evaluate_file(easy_path, "easy")
results_hard = evaluate_file(hard_path, "hard")
all_results = results_easy + results_hard

# === CSV fields ===
fieldnames = [
    "folder", "difficulty", "question",
    "context_precision", "context_recall",
    "faithfulness", "answer_relevance"
]

# === Write or append CSV ===
output_csv.parent.mkdir(parents=True, exist_ok=True)
file_exists = output_csv.exists()

with open(output_csv, "a", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for row in all_results:
        writer.writerow(row)

print(f"✅ Evaluation complete for folder '{target_dir.name}'. Results appended to {output_csv}")
