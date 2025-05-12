import json
from rag_metrics import RAGMetrics

# Load your data file
with open("data.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Init the metrics
metrics = RAGMetrics()

results = []

for item in dataset:
    question = item["question"]
    contexts = item["contexts"]
    answer = item["answer"]
    reference = item["reference"]

    row_metrics = {
        "question": question,
        "context_precision": metrics.context_precision(contexts, reference),
        "context_recall": metrics.context_recall(contexts, reference),
        "faithfulness": metrics.faithfulness(contexts, answer),
        "answer_relevance": metrics.answer_relevance(question, answer),
    }
    results.append(row_metrics)

# Print summary
for r in results:
    print(json.dumps(r, indent=2, ensure_ascii=False))

# (Optional) Save to file
with open("evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
