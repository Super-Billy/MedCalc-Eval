from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Type, Dict
from method.method import Method
from evaluator.regEvaluator import RegEvaluator
from evaluator import Evaluator
from model.model import LLM
import json
import os
import warnings
from pathlib import Path
import pandas as pd


logger = logging.getLogger(__name__)
class TwoAgent(Method):
    def __init__(
        self,
        prompt_style: str,
        llms: List[List[LLM]],
        evaluators: List[Evaluator],
        **kwargs,
    ) -> None:
        # prompt-style check 
        valid_prompt_styles = {
            "direct":   (self._direct_agent1_prompts, self._direct_agent2_prompts),
            "cot":      (self._cot_agent1_prompts,    self._cot_agent2_prompts),
            "oneshot":  (self._one_shot,              self._one_shot),
        }
        if prompt_style not in valid_prompt_styles:
            raise ValueError(f"Prompt style '{prompt_style}' not supported.")
        self.prompt_style = prompt_style
        self.agent1_prompt_fn, self.agent2_prompt_fn = valid_prompt_styles[prompt_style]

        # LLM list check 
        if not llms or any(len(pair) != 2 for pair in llms):
            raise ValueError("`llms` must be List[List[LLM]] and every inner list must contain **exactly 2** LLMs.")
        self.llm_groups = llms

        if not evaluators:
            raise ValueError("At least one Evaluator must be provided.")
        self.evaluators = evaluators

        # for build_record
        self.responses:      Dict[str, List[str]]            = {}
        self.input_tokens:   Dict[str, List[int]]            = {}
        self.output_tokens:  Dict[str, List[int]]            = {}
        self.correctness:    Dict[str, Dict[str, List[bool]]] = {}
        self.history: Dict[str, List[Dict[str, str]]] = {}

        self.batch_size = kwargs.get("batch_size", 1)
        self.use_rag    = False      # modified at generate_raw

        super().__init__(llms=llms, **kwargs)

    def generate_raw(
        self,
        test: bool = False,
        use_rag: bool = False,
        raw_json_dir: str = "raw_output",
    ) -> str:
        """Run every agent pair once; cache *responses* + *token usage*."""

        self.df = self.load_data_test() if test else self.load_dataset()

        questions       = self.df["Question"].tolist()
        patient_notes   = self.df["Patient Note"].tolist()
        calids          = self.df["Calculator ID"].astype(str).tolist()

        # -------- Agent-1 prompt --------
        prompts_agent1 = (
            self.agent1_prompt_fn(calids, patient_notes, questions)
            if self.prompt_style == "oneshot"
            else self.agent1_prompt_fn(questions)
        )


        Path(raw_json_dir).mkdir(parents=True, exist_ok=True)

        model_to_path: Dict[str, str] = {}


        for g_idx, (agent1, agent2) in enumerate(self.llm_groups):
            # ========== Agent-1 ==========
            g1 = agent1.generate(prompts_agent1, batch_size=self.batch_size)
            resp1, in1, out1 = zip(*g1)
            parsed_formula = RegEvaluator.parse_answer_agent(list(resp1))

            # ========== Agent-2 ==========
            prompts_agent2 = (
                self.agent2_prompt_fn(parsed_formula, patient_notes, questions)
                if self.prompt_style != "oneshot"
                else self.agent2_prompt_fn(calids, patient_notes, questions)
            )
            g2 = agent2.generate(prompts_agent2, batch_size=self.batch_size)
            resp2, in2, out2 = zip(*g2)

            # ========== add to record ==========
            model_name = agent1.model_name_full
            self.responses[model_name]     = list(resp2)                         # only evaluate the first response
            self.input_tokens[model_name]  = [i1 + i2 for i1, i2 in zip(in1, in2)]
            self.output_tokens[model_name] = [o1 + o2 for o1, o2 in zip(out1, out2)]
            self.correctness[model_name]   = {}

            self.history[model_name] = [
                {"agent1": r1, "agent2": r2}
                for r1, r2 in zip(list(resp1), list(resp2))
            ]

            # ========== Dump json ==========
            records = self._build_records(model_name, include_evaluation=False)
            raw_path = (
                Path(raw_json_dir) /
                f"TwoAgent_group_{g_idx}_{model_name.replace('/', '_')}_{self.prompt_style}_raw.json"
            )
            self._dump_json(records, raw_path)
            logger.info("Raw written → %s  (%d rows)", raw_path, len(records))

            model_to_path[model_name] = str(raw_path)

        if len(model_to_path) == 1:
            return next(iter(model_to_path.values()))
        return json.dumps(model_to_path, indent=2)


    def evaluate(
        self,
        raw_json_file: str,
        eval_json_dir: str = "eval_output",
    ) -> str:
        """Evaluate a cached raw file with *all* evaluators."""
        raw_path = Path(raw_json_file)
        if not raw_path.exists():
            raise FileNotFoundError(raw_path)

        with raw_path.open("r", encoding="utf-8") as f:
            records = json.load(f)
        self.df = pd.DataFrame(records)

        model_name = self.df["Model Name"].iloc[0]
        safe_name  = model_name.replace("/", "_")

        self.responses[model_name]     = self.df["LLM Original Answer"].tolist()
        self.input_tokens[model_name]  = self.df["Input Tokens"].tolist()
        self.output_tokens[model_name] = self.df["Output Tokens"].tolist()
        self.correctness[model_name]   = {}

        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids        = self.df["Calculator ID"].astype(str).tolist()
        upper_limits  = self.df["Upper Limit"].tolist()
        lower_limits  = self.df["Lower Limit"].tolist()

        # -------- iter all evaluator --------
        for ev in self.evaluators:
            if (
                "LLM_Evaluator" in ev.get_evaluator_name()
                and self.prompt_style == "direct"
            ):
                logger.debug("Skip %s on direct style", ev.get_evaluator_name())
                continue

            key, result_list = ev.check_correctness(
                responses            = self.responses[model_name],
                ground_truths        = ground_truths,
                calids               = calids,
                upper_limits         = upper_limits,
                lower_limits         = lower_limits,
                relevant_entities    = self.df.get("Relevant Entities", []).tolist(),
                ground_truth_explanations = self.df.get("Ground Truth Explanation", []).tolist(),
            )
            self.correctness[model_name][key] = result_list
            self.df[key] = result_list
            logger.info("Evaluator %s finished.", ev.get_evaluator_name())

        # -------- LLM-based evaluator token --------
        for ev in self.evaluators:
            if "LLM_Evaluator" in ev.get_evaluator_name():
                n = len(self.df)
                logger.info(
                    "### LLM Eval Avg Tokens - %s - in: %.2f | out: %.2f",
                    ev.get_evaluator_name(),
                    ev.input_token_used / n if n else 0,
                    ev.output_token_used / n if n else 0,
                )

        # -------- dump _eval.json --------
        Path(eval_json_dir).mkdir(parents=True, exist_ok=True)
        eval_path = (
            Path(eval_json_dir) /
            raw_path.name.replace("_raw.json", "_eval.json")
        )
        full_records = self._build_records(model_name, include_evaluation=True)
        self._dump_json(full_records, eval_path)
        logger.info("Eval file written → %s", eval_path)

        return str(eval_path)
    




    # -- HELPER FUNCTIONS --

    @staticmethod
    def _direct_agent1_prompts(questions: List[str]) -> List[Tuple[str, str]]:
        sys_msg = (
            "You are a medical calculation agent. Your goal is to accurately identify the appropriate "
            "medical calculation formula or scoring criteria from the user's message, "
            "and list all the required variables and recommended standard units, "
            "or all the detailed specific criteria of the scoring standard. "
            "Ensure that your knowledge and answers are completely accurate and rigorous."
        )
        prompts = []
        for q in questions:
            user_msg = (
                f"For the following medical question, determine which medical formula or scoring system should be used:\n\n"
                f"Question: {q}\n\n"
                "Include:\n"
                "1. The name and exact mathematical expression.\n"
                "2. All variables, along with recommended units or value ranges.\n"
                "3. Any conditions or considerations for proper use.\n\n"
                'Please directly output the calculation result formatted as '
                '"answer": the relevant details'
            )
            prompts.append((sys_msg, user_msg))
        return prompts

    @staticmethod
    def _direct_agent2_prompts(agent1_outputs: List[str],
                              patient_notes: List[str],
                              questions: List[str]) -> List[Tuple[str, str]]:
        sys_msg = "You are a medical calculation agent."
        prompts = []
        for formula_details, note, question in zip(agent1_outputs, patient_notes, questions):
            user_msg = (
                "Based on the question, formula and patient notes provided, please calculate the final result. "
                "Your response must be structured and clearly indicate the calculated value. "
                "Please output the result in a structured JSON format\n"
                f"Question: {question}\n"
                f"Medical Formula Details: {formula_details}\n"
                f"Patient Note: {note}\n\n"
                "Output format example:\n"
                '{"answer": <calculated value>}'
            )
            prompts.append((sys_msg, user_msg))
        return prompts

    @staticmethod
    def _cot_agent1_prompts(questions: List[str]) -> List[Tuple[str, str]]:
        sys_msg = (
            "You are a medical calculation agent. Your goal is to accurately identify the appropriate "
            "medical calculation formula or scoring criteria from the user's message, "
            "and list all the required variables and recommended standard units, "
            "or all the detailed specific criteria of the scoring standard. "
            "Ensure that your knowledge and answers are completely accurate and rigorous."
        )
        prompts = []
        for q in questions:
            user_msg = (
                f"For the following medical question, determine which medical formula or scoring system should be used:\n\n"
                f"Question: {q}\n\n"
                "Include:\n"
                "1. The name and exact mathematical expression.\n"
                "2. All variables, along with recommended units or value ranges.\n"
                "3. Any conditions or considerations for proper use.\n\n"
                "Please think step-by-step to analyze the problem and output your chain-of-thought "
                "along with the final answer.\n\n"
                'Please directly output the calculation result formatted as "answer": the relevant details'
            )
            prompts.append((sys_msg, user_msg))
        return prompts

    @staticmethod
    def _cot_agent2_prompts(agent1_outputs: List[str],
                           patient_notes: List[str],
                           questions: List[str]) -> List[Tuple[str, str]]:
        sys_msg = "You are a medical calculation agent."
        prompts = []
        for formula_details, note, question in zip(agent1_outputs, patient_notes, questions):
            user_msg = (
                "Based on the question, formula and patient notes provided, please calculate the final result. "
                "Your response must be structured and clearly indicate the calculated value. "
                "Please output the result in a structured JSON format.\n"
                "Please think step-by-step and output your chain-of-thought before presenting the final answer.\n"
                f"Question: {question}\n"
                f"Medical Formula Details: {formula_details}\n"
                f"Patient Note: {note}\n\n"
                "Output format example:\n"
                '{"step_by_step_thinking": "<your detailed reasoning>", "answer": <calculated value>}'
            )
            prompts.append((sys_msg, user_msg))
        return prompts

    @staticmethod
    def _one_shot(calids: List[str],
                 notes: List[str],
                 questions: List[str]) -> List[Tuple[str, str]]:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(
            current_dir, "..", "data", "one_shot_finalized_explanation.json"
        )
        try:
            with open(json_path, "r") as file:
                one_shot_json = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading one-shot examples: {e}")

        prompts = []
        for calid, note, question in zip(calids, notes, questions):
            if str(calid) not in one_shot_json:
                raise ValueError(f"Calculator ID {calid} not found in one-shot examples")
            example = one_shot_json[str(calid)]
            example_note = example["Patient Note"]
            example_output = {
                "step_by_step_thinking": example["Response"]["step_by_step_thinking"],
                "answer": example["Response"]["answer"]
            }

            system_msg = (
                "You are a helpful assistant for calculating a score for a given patient note. "
                "Please think step-by-step to solve the question and then generate the required score. "
                "Your output should only contain a JSON dict formatted as "
                '{"step_by_step_thought": str(your_chain_of_thought), '
                '"answer": str(your_final_answer)}.'
            )
            system_msg += f"\n\nHere is an example patient note:\n\n{example_note}"
            system_msg += f"\n\nHere is an example task:\n\n{question}"
            system_msg += (
                f'\n\nPlease directly output the JSON dict formatted as '
                f'{{"step_by_step_thought": str(...), '
                f'"answer": str(...)}}:\n\n{json.dumps(example_output)}'
            )

            user_msg = (
                f"Here is the patient note:\n\n{note}\n\n"
                f"Here is the task:\n\n{question}\n\n"
                'Please directly output your_chain_of_thought and the answer formatted as '
                '"answer": str(...)'
            )
            prompts.append((system_msg, user_msg))
        return prompts

    @staticmethod
    def _get_formula(jsonpath: str, calid: List[str]) -> List[str]:
        """
        DEPRECATED: This function is deprecated and will be removed in a future release.
        Please use the RAG-based retrieval method instead.

        Read the JSON at `jsonpath` and return a list of formulas for the given Calculator IDs.

        Args:
            jsonpath (str): Path to the JSON file.
            calid (List[str]): List of Calculator ID strings.

        Returns:
            List[str]: List of formula strings.

        Raises:
            KeyError: If a Calculator ID is not found in the JSON.
        """
        # warning for deprecation
        warnings.warn(
            "get_formula() is deprecated and will be removed in a future release; "
            "please use the RAG-based retrieval method instead.",
            category=DeprecationWarning,
            stacklevel=2
        )

        with open(jsonpath, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        formula_list = []
        for cid in calid:
            try:
                formula = json_data[cid]["Response"]["formula"]
            except KeyError:
                raise KeyError(f"Calculator ID {cid}'s formula not found in the JSON data.")
            formula_list.append(formula)

        return formula_list
