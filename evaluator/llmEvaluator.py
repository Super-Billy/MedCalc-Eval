from evaluator import RegEvaluator
from model import LLM, APIModel
from typing import List, Tuple, Optional, Type
from pydantic import BaseModel, Field
import json
from collections import OrderedDict
import csv
import logging
import numpy as np
import os
from schema.schemas import *
from pathlib import Path
import pandas as pd
from .baseEvaluator import Evaluator

logger = logging.getLogger(__name__)


class LLM_Evaluator(Evaluator):
    def __init__(self, model: LLM):
        """
        Initialize the evaluator with a given LLM instance.
        """
        self.model = model if model else APIModel("DeepSeek/deepseek-chat", temperature=0.1)
        self.input_token_used = 0
        self.output_token_used = 0

    
    def check_correctness(self, responses, ground_truths, ground_truth_explanations, relevant_entities, **kwargs):
        """
        Evaluate each structured step of an LLM response against its corresponding ground truth.

        Args:
            responses: List of LLM outputs (structured dicts or raw strings).
            ground_truths: List of ground truth final answers.
            ground_truth_explanations: List of gold-standard step-by-step explanations.
            relevant_entities: List of ground truth extracted variable values.

        Returns:
            Tuple[str, List[Dict]]: Key "LLM Evaluation", and a list of structured evaluations per response.
        """
        num_samples = len(responses)
        parsed_responses = []

        # Parse or fallback each response into structured dict format
        for resp in responses:
            if isinstance(resp, str):
                try:
                    parsed = json.loads(resp)
                except json.JSONDecodeError:
                    parsed = {}
            else:
                parsed = resp
            parsed_responses.append({
                "formula": parsed.get("formula", resp),
                "extracted_values": parsed.get("extracted_values", resp),
                "calculation": parsed.get("calculation", resp),
                "answer": parsed.get("final_answer", parsed.get("answer", resp)),
            })

        # Prepare batched prompts for each field
        all_evaluations = {field: [] for field in ["formula", "extracted_values", "calculation", "answer"]}
        for field, reference_list in [
            ("formula", ground_truth_explanations),
            ("extracted_values", relevant_entities),
            ("calculation", ground_truth_explanations),
            ("answer", ground_truths)
        ]:
            prompts = []
            for i in range(num_samples):
                system_msg, user_msg = self._gen_eval_prompt(
                    answer=parsed_responses[i][field],
                    reference=reference_list[i],
                    name_of_step=field
                )
                prompts.append((system_msg, user_msg))
            outputs = self.model.generate(prompts=prompts, schema=EvaluationAspect)

            for response, in_tokens, out_tokens in outputs:
                if isinstance(response, (str, bytes, bytearray)):
                    try:
                        parsed = json.loads(response)
                    except json.JSONDecodeError:
                        parsed = response
                else:
                    parsed = response

                all_evaluations[field].append(parsed)
                self.input_token_used += in_tokens
                self.output_token_used += out_tokens

        # Aggregate step-wise results per response
        evaluations = []
        for i in range(num_samples):
            eval_result = {
                "formula": all_evaluations["formula"][i],
                "extracted_values": all_evaluations["extracted_values"][i],
                "calculation": all_evaluations["calculation"][i],
                "answer": all_evaluations["answer"][i],
            }
            evaluations.append(eval_result)

        return "LLM Evaluation", evaluations


    def _gen_eval_prompt(self, answer, reference, name_of_step):
        system_msg = (
            "You are a medical calculation assistant. Evaluate whether a calculation step is correct by comparing it to the gold-standard reference."
        )
        user_msg = (
            f"{name_of_step.capitalize()}:\n{answer}\n\n"
            f"Gold-standard reference (fully correct):\n{reference}\n\n"
            f"Determine if the step is correct. Respond in this JSON format:\n\n"
            '{"result": "Correct" or "Incorrect", "explanation": "Brief justification."}'
        )
        if name_of_step == "formula":
            user_msg += (
                "\n\n"
                "Note: Focus only on the equivalency of the formula (equation) "
            )
        elif name_of_step == "extracted_values":
            user_msg += (
                "\n\n"
                "Note: Check if all variable given in the gold-standard reference are found or implied"
                "Ignore any naming discrepancies, as long as the meaning is the same. "
                "It is ok if the answer has more variables than the gold-standard reference. "
            )
        elif name_of_step == "calculation":
            user_msg += (
                "\n\n"
                "Note: Focus solely on whether each arithmetic step (e.g., addition, subtraction, multiplication, division, square root, etc.) is mathematically valid. "
                "If the calculated result has minor differences due to rounding or decimal approximations, consider them acceptable."
            )
        elif name_of_step == "answer" or name_of_step == "final_answer":
            user_msg += (
                "\n\n"
                "Note: If one has a unit and the other does not, please ignore the unit. "
                "If the given answer has a different unit than the gold-standard answer, please do conversion first. "
                "Allow rounding errors up to the integer part. "
            )
        return system_msg, user_msg

    
    @staticmethod
    def evaluate_formula_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the formula or scoring criteria.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution according to the given ground truth answer."
            )
            user_msg = (
                f"This is the ground truth answer:\n{ground_truth}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                "Your task is to check the formula or scoring criteria:\n"
                "Determine if the solution uses the correct medical formula(s) or scoring standard(s) as described in the ground truth answer.\n"
                "Only consider the mathematical formula functions and any related helper functions or scoring standard(s).\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_variables_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of variable substitution.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, relevant_entity in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution according to the given information."
            )
            user_msg = (
                # f"This is the ground truth answer:\n{ground_truth}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                f"Relevant Entities:\n{relevant_entity}\n\n"
                "Your task is to compare the variable values used in the solution with the values specified in the relevant entities.\n"
                "Consider ONLY the variables provided in the relevant entities, not any derived or later calculated variables in the solution.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_calculation_prompts(
        solutions: List[str],
        ground_truths: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the calculation process.
        """
        prompts = []
        for solution, ground_truth, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truths, upperlimits, lowerlimits, relevant_entities_list
        ):
            # helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the medical calculation solution."
            )
            user_msg = (
                f"This is the solution to be evaluated:\n{solution}\n\n"
                "Your task is to evaluate only the arithmetic calculation process in the solution. "
                "Ignore any issues related to the selection of formulas, substituted values, or clinical appropriateness. "
                "Focus solely on whether each arithmetic step (e.g., addition, subtraction, multiplication, division, square root, etc.) is mathematically valid. "
                "If the calculated result has minor differences due to rounding or decimal approximations, consider them acceptable. "
                "Determine if each step is mathematically correct.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation' summarizing your reasoning."
            )

            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def evaluate_final_answer_prompts(
        solutions: List[str],
        ground_truth_ans: List[str],
        upperlimits: List[str],
        lowerlimits: List[str],
        relevant_entities_list: List[str]
    ) -> List[Tuple[str, str]]:
        """
        Generate prompts for evaluating the correctness of the final answer.
        """
        prompts = []
        for solution, ground_truth_an, upperlimit, lowerlimit, _ in zip(
            solutions, ground_truth_ans, upperlimits, lowerlimits, relevant_entities_list
        ):
            helper = f"Note: The final answer WITHIN the range ({lowerlimit}, {upperlimit}) is also considered correct. You MUST be careful with numbers when comparing the final answer with the ground truth answer."
            system_msg = (
                "You are a medical calculation assistant. Your task is to evaluate the final answer in the medical solution."
            )
            user_msg = (
                f"This is the ground truth answer:\n{ground_truth_an + helper}\n"
                f"This is the solution to be evaluated:\n{solution}\n\n"
                f"Remember, the final answer should fall within the range ({lowerlimit}, {upperlimit}).\n"
                "Your task is to compare the final numerical result in the solution with the ground truth answer. Only judge whether the final numerical result is correct.\n\n"
                "Please provide your evaluation in a structured JSON format with two keys: 'result' (either 'Correct' or 'Incorrect') and a brief 'explanation'."
            )
            prompts.append((system_msg, user_msg))
        return prompts
    
    def get_evaluator_name(self) -> str:
        """
        Returns the name of the evaluator.
        """
        return f'LLM_Evaluator using {self.model.model_name_full}'
    