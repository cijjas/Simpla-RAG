import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("your_data.csv") 

df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)

metrics = [
    "context_recall", "faithfulness", "answer_relevance",
    "answer_correctness", "rouge_l", "context_precision",
    "retrieval_prec", "retrieval_rec"
]

def exp_label(exp):
    return f"{exp} (@10)" if exp in [5, 6] else str(exp)

grouped = df.groupby(["experiment", "difficulty"])[metrics].mean().reset_index()
grouped["experiment_label"] = grouped["experiment"].apply(exp_label)

def plot_global_radar(df_grouped, difficulty, metrics):
    subset = df_grouped[df_grouped["difficulty"] == difficulty]
    if subset.empty:
        return

    labels = metrics
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    palette = sns.color_palette("hsv", len(subset))

    for i, (_, row) in enumerate(subset.iterrows()):
        values = row[metrics].tolist()
        values += [values[0]]  
        label = exp_label(row["experiment"])
        ax.plot(angles, values, label=f"Exp {label}", color=palette[i])
        ax.fill(angles, values, alpha=0.1, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title(f"Radar Plot - {difficulty.capitalize()} Difficulty", size=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()

plot_global_radar(grouped, "easy", metrics)
plot_global_radar(grouped, "hard", metrics)

