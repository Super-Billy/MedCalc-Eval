import pandas as pd
from pathlib import Path

# Define paths
base_dir = Path("dataset_cleaned")
train_path = base_dir / "train_data_no_duplicate.csv"
overlap_path = base_dir / "overlap_full.csv"
overlap_notes_path = base_dir / "overlap_notes_only.csv"
output_stats_path = base_dir / "overlap_stats.txt"

# Load datasets
train_df = pd.read_csv(train_path)
overlap_df = pd.read_csv(overlap_path)
overlap_notes_df = pd.read_csv(overlap_notes_path)

# --- Full Overlap ---
total_full_overlap = len(overlap_df)
full_per_calc = overlap_df["Calculator ID"].value_counts().sort_index()
train_rows_to_remove_full = overlap_df["Train Row"].unique()
remaining_train_full = len(train_df) - len(train_rows_to_remove_full)

# --- Note-Only Overlap ---
total_note_overlap = len(overlap_notes_df)
note_per_calc = overlap_notes_df["Calculator ID"].value_counts().sort_index()
train_rows_to_remove_note = overlap_notes_df["Train Row"].unique()
remaining_train_note = len(train_df) - len(train_rows_to_remove_note)

# --- Output ---
with open(output_stats_path, "w") as f:
    f.write("=== Full Overlap (Note + Question) ===\n")
    f.write(f"Total overlaps: {total_full_overlap}\n")
    f.write("Overlaps per Calculator ID:\n")
    f.write(full_per_calc.to_string() + "\n")
    f.write(f"Unique train rows removed: {len(train_rows_to_remove_full)}\n")
    f.write(f"Remaining train rows: {remaining_train_full}\n\n")

    f.write("=== Note-Only Overlap (Any Question) ===\n")
    f.write(f"Total overlaps: {total_note_overlap}\n")
    f.write("Overlaps per Calculator ID:\n")
    f.write(note_per_calc.to_string() + "\n")
    f.write(f"Unique train rows removed: {len(train_rows_to_remove_note)}\n")
    f.write(f"Remaining train rows: {remaining_train_note}\n")

print(f"Overlap stats saved to: {output_stats_path}")
