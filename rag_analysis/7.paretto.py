import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")  
df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)


# Esto es totalmente falopa pero que se yo quizas sirve
x_metric = "context_precision"  
y_metric = "rouge_l"           

grouped = df.groupby("experiment")[[x_metric, y_metric]].mean().reset_index()

def is_dominated(point, others):
    return any(
        (other[x_metric] >= point[x_metric] and other[y_metric] > point[y_metric]) or
        (other[x_metric] > point[x_metric] and other[y_metric] >= point[y_metric])
        for _, other in others.iterrows()
    )

pareto = grouped[~grouped.apply(lambda row: is_dominated(row, grouped), axis=1)]
pareto_sorted = pareto.sort_values(by=x_metric)

plt.figure(figsize=(8, 6))
plt.scatter(grouped[x_metric], grouped[y_metric], label="All Experiments", color="gray")
plt.plot(pareto_sorted[x_metric], pareto_sorted[y_metric], marker='o', color="green", label="Pareto Frontier")

for _, row in grouped.iterrows():
    plt.text(row[x_metric] + 0.001, row[y_metric], f"Exp {int(row['experiment'])}", fontsize=9)

plt.title(f"Pareto Frontier: {x_metric} vs {y_metric}")
plt.xlabel(x_metric)
plt.ylabel(y_metric)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
