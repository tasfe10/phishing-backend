import pandas as pd

# Load English dataset
df1 = pd.read_csv("data/sms_dataset.csv")

# Ensure correct column order
df1 = df1[["text", "label"]]

# Load Bangla dataset
df2 = pd.read_csv("data/bangla_spam.csv")

# Rename columns
df2.columns = ["type", "text"]

# Convert labels: spam=1, ham=0
df2["label"] = df2["type"].map({"spam": 1, "ham": 0})

# Keep only needed columns
df2 = df2[["text", "label"]]

# Merge
df = pd.concat([df1, df2], ignore_index=True)

# Clean
df = df.dropna()
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Shuffle
df = df.sample(frac=1, random_state=42)

# Save
df.to_csv("data/final_dataset.csv", index=False)

print("✅ Fixed dataset created!")
print(df.head())
print(df["label"].value_counts())