def filter_jsonl(file_path):
        """
        Reads a JSON Lines (jsonl) file and filters records into two groups:
        
        1. Records where "Result" is "Correct" but "evaluation" -> "final_answer" -> "result" is "Incorrect".
        These row numbers are printed under the header:
        "Raw Answer Correct --------------- GPT Answer Incorrect"
        
        2. Records where "Result" is "Incorrect" but "evaluation" -> "final_answer" -> "result" is "Correct".
        These row numbers are printed under the header:
        "Raw Answer Incorrect --------------- GPT Answer Correct"
        
        :param file_path: Path to the jsonl file.
        """
        # Initialize lists to hold row numbers for each condition
        group1 = []  # Raw Answer Correct and GPT Answer Incorrect
        group2 = []  # Raw Answer Incorrect and GPT Answer Correct

        # Read the file line by line (assuming jsonl format: one JSON object per line)
        with open(file_path, 'r', encoding='utf-8') as file:
            datas = json.load(file)
            for data in datas:

                # Retrieve necessary fields safely
                result = data.get("Result")
                final_result = data.get("evaluation", {}).get("final_answer", {}).get("result")
                row_number = data.get("Row Number")

                # Append the row number to the appropriate group based on conditions
                if result == "Correct" and final_result == "Incorrect":
                    group1.append(row_number)
                elif result == "Incorrect" and final_result == "Correct":
                    group2.append(row_number)

        # Print the groups with their respective headers (header printed only once)
        if group1:
            print("Raw Answer Correct --------------- GPT Answer Incorrect")
            for row in group1:
                print(row)
        else:
            print("No records found with Raw Answer Correct and GPT Answer Incorrect")

        if group2:
            print("\nRaw Answer Incorrect --------------- GPT Answer Correct")
            for row in group2:
                print(row)
        else:
            print("No records found with Raw Answer Incorrect and GPT Answer Correct")