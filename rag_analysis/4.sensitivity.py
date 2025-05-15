import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("combined_data.csv")  

df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)

metrics = [
    "context_precision", "context_recall", "faithfulness",
    "answer_relevance", "answer_correctness", "rouge_l",
    "retrieval_prec", "retrieval_rec"
]

summary = df.groupby(["experiment", "difficulty"])[metrics].mean().reset_index()
var_per_metric = summary.groupby("difficulty")[metrics].var().T

var_per_metric.reset_index(inplace=True)
var_per_metric = var_per_metric.rename(columns={"index": "metric"})

melted = var_per_metric.melt(id_vars="metric", var_name="difficulty", value_name="variance")

melted["avg_var"] = melted.groupby("metric")["variance"].transform("mean")
melted.sort_values("avg_var", ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
sns.barplot(data=melted, x="metric", y="variance", hue="difficulty", palette="Set2")
plt.title("Metric Sensitivity Across Experiments (Variance)")
plt.ylabel("Variance Across Experiments")
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

melted.drop(columns="avg_var").to_csv("metric_variability.csv", index=False)
print("Saved metric sensitivity table to 'metric_variability.csv'")
