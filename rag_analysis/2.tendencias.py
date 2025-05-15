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

melted = df.melt(id_vars=["experiment", "difficulty"], value_vars=metrics,
                 var_name="metric", value_name="score")

summary = melted.groupby(["experiment", "difficulty", "metric"])["score"].agg(["mean", "std"]).reset_index()

g = sns.FacetGrid(summary, col="metric", col_wrap=3, sharey=False, height=4, aspect=1.3)

g.map_dataframe(
    sns.lineplot,
    x="experiment",
    y="mean",
    hue="difficulty",
    style="difficulty",
    marker="o",
    err_style=None
)

for ax, (metric, group) in zip(g.axes.flat, summary.groupby("metric")):
    for difficulty in group["difficulty"].unique():
        sub = group[group["difficulty"] == difficulty]
        ax.errorbar(
            x=sub["experiment"],
            y=sub["mean"],
            yerr=sub["std"],
            fmt="none",
            capsize=3,
            label=None,
            color=sns.color_palette()[0] if difficulty == "easy" else sns.color_palette()[1]
        )

g.set_titles("{col_name}")
g.set_axis_labels("Experiment", "Score")
g.add_legend(title="Difficulty")
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("Evolution of Performance Metrics Across Experiments")
plt.tight_layout()
plt.show()
