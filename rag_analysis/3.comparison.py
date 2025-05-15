import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

grouped = df.groupby(["experiment", "difficulty"])[metrics].mean().reset_index()

baseline = grouped[grouped["experiment"] == 1].set_index("difficulty")
others = grouped[grouped["experiment"] != 1]

deltas = []
for _, row in others.iterrows():
    exp = row["experiment"]
    diff = row["difficulty"]
    baseline_row = baseline.loc[diff]
    delta_row = {
        "experiment": exp,
        "difficulty": diff
    }
    for metric in metrics:
        delta_row[metric] = row[metric] - baseline_row[metric]
    deltas.append(delta_row)

delta_df = pd.DataFrame(deltas)

delta_df.to_csv("delta_vs_baseline.csv", index=False)
print("Saved delta comparison to 'delta_vs_baseline.csv'")
melted = delta_df.melt(id_vars=["experiment", "difficulty"], var_name="metric", value_name="delta")

g = sns.FacetGrid(melted, col="metric", col_wrap=3, sharey=False, height=4, aspect=1.3)
g.map_dataframe(sns.barplot, x="experiment", y="delta", hue="difficulty", palette="Set2", errorbar=None)
g.set_titles("{col_name}")
g.set_axis_labels("Experiment", "Δ from Baseline (Exp 1)")
g.add_legend(title="Difficulty")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Improvement Over Baseline (Experiment 1)")
plt.tight_layout()
plt.show()
