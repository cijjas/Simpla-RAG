import json
import csv
import sys
from pathlib import Path
from rag_metrics import RAGMetrics

# ╭───────────────────────────── CLI ─────────────────────────────╮
if len(sys.argv) != 2:
    print("Usage: python evaluate_custom.py <experiment_number>")
    sys.exit(1)

experiment_number = sys.argv[1]

# ╭──────────────────── localizar carpeta del experimento ───────────────────╮
base_dir = Path(".")
matching_dirs = [d for d in base_dir.iterdir()
                 if d.is_dir() and d.name.startswith(experiment_number)]

if not matching_dirs:
    print(f"❌ No directory found starting with '{experiment_number}'")
    sys.exit(1)

target_dir = matching_dirs[0]                       
experiment_id = target_dir.name.split(".")[0]       

easy_path = target_dir / "ragas_easy.json"
hard_path = target_dir / "ragas_hard.json"
output_csv = Path("results_custom.csv")

# ╭──────────────────── inicializar motor de métricas ───────────────────────╮
metrics = RAGMetrics()
K = 10               # @k para métricas de retrieval
THRESH = 0.80        # umbral de similitud para relevancia de chunks

# ╭────────────────────────── evaluación núcleo ─────────────────────────────╮
def evaluate_file(json_path: Path, difficulty_tag: str):
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    for item in dataset:
        question   = item["question"]
        contexts   = item["contexts"]
        answer     = item["answer"]
        reference  = item["reference"]     # ground‑truth humano

        row_metrics = {
            "experiment"            : experiment_id,
            "difficulty"            : difficulty_tag,
            "question"              : question,

            # ─ context metrics
            "context_precision"     : metrics.context_precision(
                                          contexts, reference, threshold=THRESH),

            f"retrieval_prec@{K}"   : metrics.retrieval_precision_at_k(
                                          contexts, reference, k=K, threshold=THRESH),
            f"retrieval_rec@{K}"    : metrics.retrieval_recall_at_k(
                                          contexts, reference, k=K, threshold=THRESH),
        }
        results.append(row_metrics)
    return results

# ╭────────────────────────── correr evaluación ─────────────────────────────╮
all_results = []
if easy_path.exists():
    all_results += evaluate_file(easy_path, "easy")
if hard_path.exists():
    all_results += evaluate_file(hard_path, "hard")

# ╭───────────────────────────── CSV output ─────────────────────────────────╮
fieldnames = [
    "experiment", "difficulty", "question",
    "context_precision", "context_recall",
    "faithfulness", "answer_relevance",
    "answer_correctness", f"retrieval_prec@{K}",
    f"retrieval_rec@{K}", "rouge_l"
]

file_exists = output_csv.exists()
with open(output_csv, "a", encoding="utf-8", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerows(all_results)

print(f"✅ Evaluation complete for experiment '{experiment_id}'. "
      f"Results appended to {output_csv}")
