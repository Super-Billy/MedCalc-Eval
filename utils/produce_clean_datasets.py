import pandas as pd
from pathlib import Path

# Define the input and output paths
input_path = Path("dataset/train_data.csv")
output_dir = Path("dataset_cleaned")
output_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
output_path = output_dir / "train_data_no_duplicate.csv"

# Load the dataset
df = pd.read_csv(input_path)

# Remove duplicate rows based on 'Patient Note' and 'Question'
df_deduplicated = df.drop_duplicates(subset=["Patient Note", "Question"], keep='first').reset_index(drop=True)

# Reassign Row Number starting from 1
df_deduplicated["Row Number"] = range(1, len(df_deduplicated) + 1)

# Save the cleaned dataset
df_deduplicated.to_csv(output_path, index=False)

print(f"Original dataset had {len(df)} rows.")
print(f"Cleaned dataset has {len(df_deduplicated)} rows.")
print(f"Cleaned dataset saved to: {output_path}")
