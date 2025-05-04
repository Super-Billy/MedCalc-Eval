import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path("dataset_cleaned")
train_path = base_dir / "train_data_no_duplicate.csv"
test_path = base_dir / "test_data_adjusted.csv"
overlap_output = base_dir / "overlap_full.csv"
overlap_notes_output = base_dir / "overlap_notes_only.csv"

# Load datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# === Full match on Patient Note + Question ===
full_overlap = pd.merge(
    test_df,
    train_df,
    on=["Patient Note", "Question"],
    suffixes=("_test", "_train")
)

full_overlap_result = full_overlap[["Row Number_test", "Row Number_train", "Calculator ID_test"]]
full_overlap_result.columns = ["Test Row", "Train Row", "Calculator ID"]
full_overlap_result.to_csv(overlap_output, index=False)

# === Match on Patient Note only (including full overlaps) ===
note_overlap = pd.merge(
    test_df,
    train_df,
    on="Patient Note",
    suffixes=("_test", "_train")
)

note_overlap_result = note_overlap[["Row Number_test", "Row Number_train", "Calculator ID_test"]]
note_overlap_result.columns = ["Test Row", "Train Row", "Calculator ID"]
note_overlap_result.to_csv(overlap_notes_output, index=False)

print(f"Full question+note overlaps saved to: {overlap_output}")
print(f"All note overlaps (regardless of question) saved to: {overlap_notes_output}")
