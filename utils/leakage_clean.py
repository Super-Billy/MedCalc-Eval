import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path("dataset_cleaned")
train_path = base_dir / "train_data_no_duplicate.csv"
overlap_notes_path = base_dir / "overlap_notes_only.csv"
output_path = base_dir / "train_data_no_leakage.csv"

# Load datasets
train_df = pd.read_csv(train_path)
overlap_notes_df = pd.read_csv(overlap_notes_path)

# Get unique train row numbers to remove
train_rows_to_remove = overlap_notes_df["Train Row"].unique()

# Remove rows from train set
cleaned_train_df = train_df[~train_df["Row Number"].isin(train_rows_to_remove)].reset_index(drop=True)

# Renumber the Row Number column
cleaned_train_df["Row Number"] = range(1, len(cleaned_train_df) + 1)

# Save the cleaned dataset
cleaned_train_df.to_csv(output_path, index=False)

print(f"Removed {len(train_rows_to_remove)} rows from training data.")
print(f"Cleaned training data saved to: {output_path}")
