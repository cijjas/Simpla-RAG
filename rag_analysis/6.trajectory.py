import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_data.csv")

df.rename(columns={
    "retrieval_prec@5": "retrieval_prec",
    "retrieval_rec@5": "retrieval_rec"
}, inplace=True)

x_metric = "answer_correctness"
y_metric = "faithfulness"

grouped = df.groupby("experiment")[[x_metric, y_metric]].mean().reset_index()

plt.figure(figsize=(8, 6))
plt.plot(grouped[x_metric], grouped[y_metric], marker='o', linestyle='-')

for i, row in grouped.iterrows():
    label = f"Exp {int(row['experiment'])}"
    plt.text(row[x_metric] + 0.001, row[y_metric], label)

plt.title(f"Experiment Trajectory: {x_metric} vs {y_metric}")
plt.xlabel(x_metric)
plt.ylabel(y_metric)
plt.grid(True)
plt.tight_layout()
plt.show()
