import pandas as pd
from pathlib import Path

# Define path
base_dir = Path("dataset")
test_path = base_dir / "train_data.csv"
duplicate_output = base_dir / "train_duplicates.csv"

# Load dataset
test_df = pd.read_csv(test_path)

# Check for duplicates based on Patient Note + Question
duplicates = test_df[test_df.duplicated(subset=["Patient Note", "Question"], keep=False)]

# Save the duplicate entries to CSV
duplicates.to_csv(duplicate_output, index=False)

print(f"Duplicate rows (same Patient Note + Question) saved to: {duplicate_output}")
print(f"Total duplicates found: {len(duplicates)}")