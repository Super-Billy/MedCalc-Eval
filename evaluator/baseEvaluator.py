import os
import json
import re
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)

class Evaluator(ABC):
    
    @abstractmethod
    def check_correctness(
        responses: List[str], 
        ground_truth: List[str], 
        calid: List[Union[str, int]],
        upper_limit: List[Union[str, float]], 
        lower_limit: List[Union[str, float]],
        ground_truth_explanations: List[str], 
        relevant_entities: List[Union[dict, str]],
    ) -> Tuple[str, List[str]]:
        """
        Check correctness of answers against ground truth.
        
        Args:
            responses: List of LLM responses
            ground_truth: List of ground truth answers
            calid: List of calculator IDs for each answer
            upper_limit: Upper bounds for acceptable answers
            lower_limit: Lower bounds for acceptable answers
            ground_truth_explanations: List of explanations for the ground truth
            relevant_entities: List of relevant entities for each answer
            
        Returns:
            Tuple with key "Result" and list of "Correct"/"Incorrect" values.
        """
        pass


    @staticmethod
    def compute_overall_accuracy_new(input_file_path: str, output_dir_path: str):
        file_name = os.path.basename(input_file_path)
        base_name = os.path.splitext(file_name)[0]
        os.makedirs(output_dir_path, exist_ok=True)
        ext = Path(input_file_path).suffix.lower()

        # Load file
        datas = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            if ext == '.json':
                datas = json.load(f)
            elif ext == '.jsonl':
                for line in f:
                    line = line.strip()
                    if line:
                        datas.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # Accumulators
        regular_eval = defaultdict(list)
        llm_eval = defaultdict(lambda: {
            "results": [],
            "first_error_type": [],
            "error_type_counts": Counter()
        })
        input_tokens = []
        output_tokens = []

        # Fields to check in LLM Evaluation
        fields = ["formula", "extracted_values", "calculation", "answer"]

        for data in datas:
            category = data.get("Category", "Unknown")
            result = str(data.get("Result"))
            input_tokens.append(data.get("Input Tokens", -1))
            output_tokens.append(data.get("Output Tokens", -1))

            # Regular correctness
            is_correct = 1 if result in ["1", "Correct"] else 0
            regular_eval[category].append(is_correct)

            # LLM Evaluation correctness
            if "LLM Evaluation" in data:
                llm_res = data["LLM Evaluation"]
                bools = [
                    llm_res.get(f, {}).get("result", "") == "Correct"
                    for f in fields
                ]
                all_correct = all(bools)
                llm_eval[category]["results"].append(1 if all_correct else 0)

                if not all_correct:
                    # record first error type
                    for f, ok in zip(fields, bools):
                        if not ok:
                            llm_eval[category]["first_error_type"].append(f)
                            llm_eval[category]["error_type_counts"][f] += 1
                            break

        # helper to calc avg, std (of mean), count
        def calc_stats(scores: list):
            n = len(scores)
            if n == 0:
                return {"average": 0.0, "std": 0.0, "count": 0}
            arr = np.array(scores, dtype=float)
            mean = arr.mean()
            # standard error of proportion
            std_err = np.sqrt(mean * (1 - mean) / n)
            return {
                "average": round(mean * 100, 2),
                "std": round(std_err, 2),
                "count": n
            }

        output = {}
        # per-category
        for category in sorted(set(regular_eval) | set(llm_eval)):
            reg_scores = regular_eval[category]
            llm_scores = llm_eval[category]["results"]

            output[category] = {
                "regular expression evaluation": calc_stats(reg_scores),
                "llm evaluation": calc_stats(llm_scores)
            }

            # breakdown of error types
            incorrect = llm_scores.count(0)
            if incorrect:
                counts = llm_eval[category]["error_type_counts"]
                for f in fields:
                    output[category]["llm evaluation"][f"{f} error"] = \
                        round((counts[f] / incorrect) * 100, 2)

        # overall aggregation
        all_reg = [v for scores in regular_eval.values() for v in scores]
        all_llm = [v for info in llm_eval.values() for v in info["results"]]

        output["overall"] = {
            "regular expression evaluation": calc_stats(all_reg),
            "llm evaluation": calc_stats(all_llm)
        }

        # overall error breakdown
        total_incorrect = all_llm.count(0)
        if total_incorrect:
            overall_counts = Counter()
            for info in llm_eval.values():
                overall_counts.update(info["error_type_counts"])
            for f in fields:
                output["overall"]["llm evaluation"][f"{f} error"] = \
                    round((overall_counts[f] / total_incorrect) * 100, 2)

        # token stats
        output["input_tokens_average"]  = int(round(np.mean(input_tokens)))  if input_tokens  else 0
        output["output_tokens_average"] = int(round(np.mean(output_tokens))) if output_tokens else 0

        # Save results
        out_file = os.path.join(output_dir_path, f"results_{base_name}.json")
        with open(out_file, 'w', encoding='utf-8') as wf:
            json.dump(output, wf, indent=4, ensure_ascii=False)

        return out_file, output


    
    @staticmethod
    def compute_multifile_overall_accuracy_new(input_dir_path: str, output_dir_path: str):
        """
        Compute accuracy for all JSON files in a directory using the updated Evaluator.compute_overall_accuracy_new.

        Args:
            input_dir_path: Directory containing the JSON files with evaluation results.
            output_dir_path: Directory where results will be written, each file will be named "results_<input_filename>.json".
        """
        os.makedirs(output_dir_path, exist_ok=True)

        # Get all JSON files in the directory
        input_files = [f for f in os.listdir(input_dir_path) if f.endswith('.json')]

        if not input_files:
            print(f"No JSON files found in directory: {input_dir_path}")
            return

        processed_files = 0
        for file_name in input_files:
            input_file_path = os.path.join(input_dir_path, file_name)

            try:
                Evaluator.compute_overall_accuracy_new(input_file_path, output_dir_path)
                processed_files += 1
                print(f"Processed file {processed_files}/{len(input_files)}: {file_name}")
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")

        print(f"Completed processing {processed_files} out of {len(input_files)} files.")


    
    