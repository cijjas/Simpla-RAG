import pandas as pd


# Helper para manejar archivos CSV de resultados de evaluación
df1 = pd.read_csv("no_absolutes.csv")
df2 = pd.read_csv("pre_absolutes.csv")

merge_keys = ["experiment", "difficulty", "question"]

merged_df = pd.merge(df1, df2, on=merge_keys, how="outer")

merged_df.to_csv("merged_results.csv", index=False)
