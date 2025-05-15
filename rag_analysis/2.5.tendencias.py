import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load and rename columns
df = pd.read_csv("combined_data.csv")
df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)

df['experiment'] = df['experiment'].apply(lambda x: f"Exp {x}")

metrics = [
    "context_precision", "context_recall", "faithfulness",
    "answer_relevance", "answer_correctness",
    "retrieval_prec", "retrieval_rec"
]

melted = df.melt(id_vars=["experiment"], value_vars=metrics,
                 var_name="metric", value_name="score")

summary = melted.groupby(["experiment", "metric"])["score"].agg(["mean", "std"]).reset_index()

for metric in metrics:
    metric_data = summary[summary["metric"] == metric]
    plt.figure(figsize=(8, 4))
    sns.lineplot(data=metric_data, x="experiment", y="mean", marker="o", linewidth=3)
    plt.xlabel("Experiment")
    plt.ylabel("Score")
    plt.ylim(0.6, 1.0)
    plt.grid(True, axis='y') 
    plt.grid(False, axis='x') 
    plt.tight_layout()
    plt.show()