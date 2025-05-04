import pandas as pd
import json

# File paths
modular_cot_path = "raw_output/modular_cot/OpenAI_gpt-4o-mini_modular_cot.jsonl"
threeagent_path = "raw_output/ThreeAgent_group_0_gpt-4o-mini_direct.jsonl"

# Load JSONL
def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

# Normalize the result to 0 or 1
def normalize_result(val):
    val_str = str(val).strip().lower()
    if val_str in ["1", "correct"]:
        return 1
    elif val_str in ["0", "incorrect"]:
        return 0
    else:
        return None

# Load and normalize data
modular_data = load_jsonl(modular_cot_path)
threeagent_data = load_jsonl(threeagent_path)

modular_df = pd.DataFrame(modular_data).set_index("Row Number")
threeagent_df = pd.DataFrame(threeagent_data).set_index("Row Number")

# Cast index to string to ensure match
modular_df.index = modular_df.index.astype(str)
threeagent_df.index = threeagent_df.index.astype(str)

# Normalize 'Result' columns
modular_df["Result"] = modular_df["Result"].apply(normalize_result)
threeagent_df["Result"] = threeagent_df["Result"].apply(normalize_result)

# Intersect row numbers
common_rows = modular_df.index.intersection(threeagent_df.index)
print(f"Modular rows: {len(modular_df)}, ThreeAgent rows: {len(threeagent_df)}, Common rows: {len(common_rows)}")

# Buckets
both_correct = []
modular_correct_threeagent_wrong = []
threeagent_correct_modular_wrong = []

# Compare normalized results
for row in common_rows:
    m_result = modular_df.loc[row, "Result"]
    t_result = threeagent_df.loc[row, "Result"]

    if m_result == 1 and t_result == 1:
        both_correct.append(row)
    elif m_result == 1 and t_result == 0:
        modular_correct_threeagent_wrong.append(row)
    elif m_result == 0 and t_result == 1:
        threeagent_correct_modular_wrong.append(row)

# Write results
with open("model_comparison_result.txt", "w") as f:
    f.write("=== Both Correct ===\n")
    for row in both_correct:
        f.write(f"{row}\n")

    f.write("\n=== Modular COT Correct, ThreeAgent Wrong ===\n")
    for row in modular_correct_threeagent_wrong:
        f.write(f"{row}\n")

    f.write("\n=== ThreeAgent Correct, Modular COT Wrong ===\n")
    for row in threeagent_correct_modular_wrong:
        f.write(f"{row}\n")

print("Comparison written to model_comparison_result.txt")
