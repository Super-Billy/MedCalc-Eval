import json
import os
import csv

def extract_formulas_from_json(json_path: str) -> str:
    """
    Read a JSON file whose top-level structure is a dict mapping arbitrary keys
    to objects, extract each object’s Response["formula"] text, and write them to
    a .txt file—wrapping each formula in markers and placing one per line.

    Args:
        json_path: Path to the input JSON file. The JSON must be an object (dict),
                   where each value is an object containing Response["formula"].

    Returns:
        The path to the generated .txt file.
    """
    # 1. Load JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        content = json.load(f)

    # 2. Ensure the top-level structure is a dict
    if not isinstance(content, dict):
        raise ValueError(f"Expected top-level JSON to be a dict, got {type(content)}")

    # 3. Prepare the output file path
    base, _ = os.path.splitext(json_path)
    output_path = f"{base}_formulas.txt"

    # 4. Iterate over all values and extract the formula text
    with open(output_path, 'w', encoding='utf-8') as out:
        for item in content.values():
            formula_text = ''
            if isinstance(item, dict):
                formula_text = item.get('Response', {}).get('formula', '')
            # Wrap each formula in markers, one per line, no extra blank lines
            out.write(f"<<FORMULA START>>{formula_text}<<FORMULA END>>\n")

    return output_path


def write_calculator_data(
    csv_path: str,
    response_json_path: str,
    output_json_path: str = "rag_check.json"
) -> None:
    """
    Read a CSV file and a JSON file, then write a combined JSON output listing
    each unique Calculator ID with its Question and Formula.

    Args:
        csv_path: Path to the input CSV file containing "Calculator ID" and "Question" columns.
        response_json_path: Path to the input JSON file mapping each Calculator ID to its Response dict.
        output_json_path: Path where the combined JSON list will be written.
    """
    # 1. Extract the first Question for each Calculator ID from the CSV
    unique_questions = {}
    with open(csv_path, mode='r', encoding='utf-8', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        if 'Calculator ID' not in reader.fieldnames or 'Question' not in reader.fieldnames:
            raise KeyError("CSV must contain 'Calculator ID' and 'Question' columns.")
        for row in reader:
            calc_id = row['Calculator ID']
            question = row['Question']
            if calc_id not in unique_questions:
                unique_questions[calc_id] = question.strip()

    # 2. Load the response JSON to get each Calculator ID's formula
    with open(response_json_path, mode='r', encoding='utf-8') as jf:
        response_data = json.load(jf)

    # 3. Build the output list: each entry contains ID, Question, and Formula
    output_list = []
    for calc_id, question in unique_questions.items():
        formula = ""
        entry = response_data.get(calc_id)
        if isinstance(entry, dict):
            resp = entry.get("Response")
            if isinstance(resp, dict):
                formula = resp.get("formula", "").strip()
        output_list.append({
            "Calculator ID": calc_id,
            "Question": question,
            "Formula": formula
        })

    # 4. Write the combined list to a new JSON file
    with open(output_json_path, mode='w', encoding='utf-8') as outjf:
        json.dump(output_list, outjf, ensure_ascii=False, indent=2)


# Usage example
if __name__ == "__main__":
    csv_input = "dataset_cleaned/test_data_adjusted.csv"
    response_json = "data/one_shot_finalized_explanation.json"
    output_file = "combined_output.json"

    write_calculator_data(csv_input, response_json, output_file)
    print(f"Combined JSON written to {output_file}")
