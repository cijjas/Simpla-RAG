import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results_ragas_1.csv")

df['experiment'] = df['experiment'].str.replace('experiment_', 'Exp ')
metrics = ['context_precision', 'context_recall', 'faithfulness']

df_avg = df.groupby('experiment')[metrics].mean().reset_index()

for metric in metrics:
    plt.figure(figsize=(8, 4))  
    plt.plot(df_avg['experiment'], df_avg[metric], marker='o', markersize=8, linewidth=3)
    plt.xlabel('Experiment')
    plt.ylabel('Score')
    plt.ylim(0.6, 1.0)
    plt.grid(True, axis='y')  
    plt.grid(False, axis='x') 
    plt.xticks(rotation=0)
    plt.tight_layout(pad=0.5)  
    plt.subplots_adjust(bottom=0.05)  
    plt.show()