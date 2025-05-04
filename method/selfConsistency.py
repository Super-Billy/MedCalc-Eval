from method import Method
from model import LLM
from evaluator import RegEvaluator
from typing import List, Tuple
from schema.schemas import prompt_style_to_schema
import os
import json
from collections import Counter
import numpy as np
from collections import defaultdict

class SelfConsistency(Method):
    def __init__(self, prompt_style: str, llms: List[LLM], evaluator: RegEvaluator, voting_times: int, **kwargs):
        '''
        Initialize a selfConsistency method instance.
        Args:
            prompt_style: The style of prompting to use.
                Must be one of: ['direct', 'cot', 'one_shot', 'modular', 'modular_cot'].
            llm_names: A list of model names (including company name) to use with this method.
        '''
        valid_prompt_styles = {
            'direct': self.direct,
            'cot': self.cot,
            'one_shot': self.one_shot,
            'modular': self.modular,
        }
        if prompt_style not in valid_prompt_styles:
            raise ValueError(f"Prompt style: {prompt_style} not supported.")
        
        self.prompt_style = prompt_style
        self.prompt_fn = valid_prompt_styles[prompt_style]
        self.evaluator = evaluator
        self.voting_times = voting_times
        self.llm_list = llms

        # Instance attributes to store evaluation results
        self.responses = {}  # Dictionary to store responses by model name
        self.answers = {}    # Dictionary to store extracted answers by model name
        self.correctness = {} # Dictionary to store correctness results by model name
        self.input_tokens = {}  # Dictionary to store token counts by model name
        self.output_tokens = {} # Dictionary to store output token counts by model name

        self.paths_by_model = {} # Dictionary to store paths to output files by model name

        super().__init__(llms=llms, **kwargs)

    def evaluate(self, test=False):
        # Load dataset
        if test:
            self.df = self.load_data_test()
        else:
            self.df = self.load_dataset()
        
        # Extract fields
        notes = self.df["Patient Note"].tolist()
        questions = self.df["Question"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()  # convert to string if needed
        ground_truths = self.df["Ground Truth Answer"].tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()

        # Make Prompts
        prompts = self.prompt_fn(calids, notes, questions)

        # Prompt model
        for llm in self.llm_list:

            model_name = llm.get_model_name()

            for i in range(self.voting_times):

                key = f"{model_name}_voting{i+1}"
                
                # Generate responses
                self.responses[key], self.input_tokens[key], self.output_tokens[key] = llm.generate(
                    prompts, 
                    schema=prompt_style_to_schema(self.prompt_style), 
                    batch_size=self.batch_size
                )
            
                # Process responses
                self.answers[key] = self.evaluator.extract_answer(self.responses[key],calid=calids)
                binary_results = self.evaluator.check_correctness(self.answers[key], ground_truths, calids, upper_limits, lower_limits)
                self.correctness[key] = ["Correct" if val == 1 else "Incorrect" for val in binary_results]

    def write_to_file(self, model_name: str, filepath: str='json_output') -> str:
        '''
        Write evaluation results to a JSONL file using stored instance attributes.
        
        Args:
            model_name: Name of the model being evaluated
            filepath: Directory to save the output file (default: 'json_output')
            
        Returns:
            str: Path to the created output file
        '''
        
        # Verify that we have results for this model
        if not any(key.startswith(model_name) for key in self.responses):
            raise ValueError(f"No stored responses found for model: {model_name}")
        
        # Create directory if it doesn't exist
        os.makedirs(filepath, exist_ok=True)
        
        output_files = []
        
        for i in range(self.voting_times):
            # Create fields dictionary from instance attributes
            fields = {
                "LLM Original Answer": self.responses[f"{model_name}_voting{i+1}"],
                "LLM Answer": self.answers[f"{model_name}_voting{i+1}"],
                "Result": self.correctness[f"{model_name}_voting{i+1}"],
                "Input Tokens": self.input_tokens[f"{model_name}_voting{i+1}"],
                "Output Tokens": self.output_tokens[f"{model_name}_voting{i+1}"],
            }
            
            # Validate that all fields have the correct length
            for field_values in fields.values():
                if len(field_values) != len(self.df):
                    raise ValueError(f"Expected all field values to have length {len(self.df)}, but got {len(field_values)}")
            
            # Replace slashes in model name with underscores
            safe_model_name = model_name.replace("/", "_")
            output_file = os.path.join(filepath, f"{safe_model_name}_{self.prompt_style}_voting{i+1}.jsonl")
            output_files.append(output_file)
            
            # Create list of records to write
            records = []
            for i in range(len(self.df)):
                row = self.df.iloc[i]
                
                # Extract base fields from DataFrame
                record = {
                    "Row Number": row["Row Number"],
                    "Calculator Name": row["Calculator Name"],
                    "Calculator ID": row["Calculator ID"],
                    "Category": row["Category"],
                    "Note ID": row["Note ID"],
                    "Patient Note": row["Patient Note"],
                    "Question": row["Question"],
                    "Ground Truth Answer": row["Ground Truth Answer"],
                    "Ground Truth Explanation": row["Ground Truth Explanation"],
                    "Upper Limit": row["Upper Limit"],
                    "Lower Limit": row["Lower Limit"],
                }
                
                # Add fields from the results
                for key, values in fields.items():
                    record[key] = values[i]
                
                records.append(record)
            
            # Write records to JSONL file
            with open(output_file, 'w') as f:
                for record in records:
                    f.write(json.dumps(record,default=str) + '\n')
            
            print(f"Results written to {output_file}")
        return output_files


    def calculate_voting_accuracy(self, model_name: str, output_dir: str):
        '''
        Calculate the voting accuracy of all the models, and write the results under the specified output directory.
        We mark an answer as correct if the majority answer among the voting times is correct.

        Args:
            output_dir: The directory to write the voting accuracy results to
        '''
        calids = self.df["Calculator ID"].astype(str).tolist()
        categories = self.df["Category"].tolist()
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()
        ground_truths = self.df["Ground Truth Answer"].tolist()

        for llm in self.llm_list:
            model_name = llm.get_model_name()
            safe_model_name = model_name.replace("/", "_")

            # Gather all answers from voting
            all_answers = []
            all_input_tokens = []
            all_output_tokens = []
            for i in range(self.voting_times):
                answer = self.answers[f"{model_name}_voting{i+1}"]
                all_answers.append(answer)
                input_tokens = self.input_tokens[f"{model_name}_voting{i+1}"]
                all_input_tokens.append(input_tokens)
                output_tokens = self.output_tokens[f"{model_name}_voting{i+1}"]
                all_output_tokens.append(output_tokens)

            # Majority vote
            transposed = list(zip(*all_answers))
            final_answers = [Counter(ans).most_common(1)[0][0] for ans in transposed]

            # Evaluate correctness
            correct_flags = self.evaluator.check_correctness(
                final_answers,
                ground_truths,
                calids,
                upper_limits,
                lower_limits
            )

            # Initialize structures for accuracy
            category_scores = defaultdict(list)
            overall_scores = []

            for correct, category in zip(correct_flags, categories):
                score = int(correct)
                category_scores[category].append(score)
                overall_scores.append(score)

            # Compute accuracy stats
            results = {}

            for cat, scores in category_scores.items():
                arr = np.array(scores)
                p = np.mean(arr)
                std = np.sqrt(p * (1 - p) / len(arr)) if len(arr) > 0 else 0.0
                results[cat] = {
                    "average": round(p * 100, 2),
                    "std": round(std, 2),
                    "count": len(arr)
                }

            # Overall accuracy
            overall_arr = np.array(overall_scores)
            p = np.mean(overall_arr)
            std = np.sqrt(p * (1 - p) / len(overall_arr)) if len(overall_arr) > 0 else 0.0
            results["overall"] = {
                "average": round(p * 100, 2),
                "std": round(std, 2),
                "count": len(overall_arr)
            }

            # Mean token usage
            results["input_tokens_average"] = int(round(np.mean(np.concatenate(all_input_tokens))))
            results["output_tokens_average"] = int(round(np.mean(np.concatenate(all_output_tokens))))

            # Save to file
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir,
                f"{safe_model_name}_{self.prompt_style}_voting_{self.voting_times}times.jsonl"
            )
            with open(output_path, "w") as f:
                f.write(json.dumps(results, indent=4) + "\n")


        

   