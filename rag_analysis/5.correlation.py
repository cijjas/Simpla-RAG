import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")
df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)

metrics = [
    "context_precision", "context_recall", "faithfulness",
    "answer_relevance", "answer_correctness", "rouge_l",
    "retrieval_prec", "retrieval_rec"
]
agg = df.groupby(["experiment", "difficulty"])[metrics].mean().reset_index()

corr = agg[metrics].corr()

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Metric Correlation Heatmap")
plt.tight_layout()
plt.show()
