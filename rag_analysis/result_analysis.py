import pandas as pd

# Load both CSV files
df1 = pd.read_csv("no_absolutes.csv")
df2 = pd.read_csv("pre_absolutes.csv")

# Define the merge keys
merge_keys = ["experiment", "difficulty", "question"]

# Perform the outer join
merged_df = pd.merge(df1, df2, on=merge_keys, how="outer")

# Save the result
merged_df.to_csv("merged_results.csv", index=False)
