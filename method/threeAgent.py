from typing import List, Optional, Tuple
from method.method import Method
from evaluator.regEvaluator import RegEvaluator
from model.model import LLM
import json
import os

class ThreeAgent(Method):
    def __init__(self, prompt_style: str, llms: List[List[LLM]], evaluator: RegEvaluator, **kwargs):
        """
        Initialize a ThreeAgent pipeline instance.

        Args:
            prompt_style: The style of prompting to use. Currently, only 'direct' is supported.
            llms: A list of groups, where each group is a list of three LLM instances.
                  (Each group corresponds to one set of three agents.)
            evaluator: An RegEvaluator instance for processing outputs.
            kwargs: Additional keyword arguments.
        """
        if not llms or len(llms) == 0:
            raise ValueError("At least one group of LLMs must be provided.")
        for group in llms:
            if len(group) != 3:
                raise ValueError("Each group must contain exactly 3 LLM instances.")
        self.llm_groups = llms
        if prompt_style != 'direct':
            raise ValueError("ThreeAgent currently supports only the 'direct' prompt style.")
        self.prompt_style = prompt_style
        self.evaluator = evaluator
        self.batch_size = kwargs.get("batch_size", 1)
        
        # DataFrame will be loaded later in evaluate().
        self.df = None
        
        # Call parent's __init__ if needed.
        super().__init__(llms=llms, **kwargs)
    
    def direct_agent1_prompts(self, questions: List[str]) -> List[Tuple[str, str]]:
        """
        Build chat prompts for Agent 1 based on the question.

        Returns:
            List of tuples (system_message, user_message).
        """
        sys_msg = (
            "You are a medical calculation agent. Your goal is to accurately identify the appropriate medical calculation formula or scoring criteria from the user's message, "
            "and list all the required variables and recommended standard units, or all the detailed specific criteria of the scoring standard. "
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
                "3. Any conditions or considerations for proper use."

                'Please directly output the calculation result formatted as '
                '"answer": the relevant details'
            )
            prompts.append((sys_msg, user_msg))
        return prompts
    
    def direct_agent2_prompts(self, agent1_outputs: List[str], patient_notes: List[str]) -> List[Tuple[str, str]]:
        """
        Build chat prompts for Agent 2 based on Agent 1's outputs and the patient note.

        Returns:
            List of tuples (system_message, user_message).
        """
        sys_msg = (
            "You are a medical analysis agent. Your goal is to read the patient's notes and the detailed information of the medical formula. "
            "Please extract from the patient notes all the variable values related to the medical formula calculation, and ensure that the units or formats match the formula requirements."
        )
        prompts = []
        for output, note in zip(agent1_outputs, patient_notes):
            user_msg = (
                "Below are the patient note and detailed information regarding the medical formula. Please extract the relevant details from the patient record:\n\n"
                f"Patient Note: {note}\n"
                f"Medical Formula: {output}"
                'Please directly output the calculation result formatted as '
                '"answer": the relevant variable details'
            )
            prompts.append((sys_msg, user_msg))
        return prompts
    
    def direct_agent3_prompts(self, agent1_outputs: List[str], agent2_outputs: List[str], questions: List[str]) -> List[Tuple[str, str]]:
        """
        Build chat prompts for Agent 3 based on Agent 1's and Agent 2's outputs.

        Returns:
            List of tuples (system_message, user_message).
        """
        sys_msg = (
            "You are a medical calculation agent. Using the given formula and relevant values, perform the required calculations accurately to answer the question."
        )
        prompts = []
        for output1, output2, question in zip(agent1_outputs, agent2_outputs, questions):
            user_msg = (
                "Please use the following formula and data to calculate the final result, answering the question:\n\n"
                f"Qestion: {question}\n"
                f"Formula: {output1}\n"
                f"Extracted Data (with units): {output2}\n\n"
                'Please directly output the calculation result formatted as '
                '"answer": answer to the question'
            )
            prompts.append((sys_msg, user_msg))
        return prompts
    
    def evaluate(self, test: bool = False, n_data: Optional[int] = 5):
        """
        Evaluate the three-agent pipeline on the dataset for each provided LLM group.

        Args:
            test (bool, optional): If True, load a test dataset. Defaults to False.
        """
        # Load dataset
        if test:
            self.df = self.load_data_test(n_data=n_data)
        else:
            self.df = self.load_dataset()
        
        # Extract required fields from the dataset.
        questions = self.df["Question"].tolist()
        patient_notes = self.df["Patient Note"].tolist()
        ground_truths = self.df["Ground Truth Answer"].tolist()
        calids = self.df["Calculator ID"].astype(str).tolist()  # used by evaluator if needed
        upper_limits = self.df["Upper Limit"].tolist()
        lower_limits = self.df["Lower Limit"].tolist()
        
        # Build prompts common to all groups.
        prompts_agent1 = self.direct_agent1_prompts(questions)
        
        # Iterate over each group of LLMs.
        for group_index, group in enumerate(self.llm_groups):
            agent1, agent2, agent3 = group[0], group[1], group[2]
            
            # --- Agent 1 Processing ---
            responses_agent1 = agent1.generate(prompts=prompts_agent1, batch_size=self.batch_size)
            tokens_agent1 = [agent1.compute_tokens(resp) for resp in responses_agent1]
            raw_agent1 = RegEvaluator.parse_deepseek(responses_agent1)
            parsed_agent1 = RegEvaluator.extract_answer_multiAgent(raw_agent1)

            # --- Agent 2 Processing ---
            prompts_agent2 = self.direct_agent2_prompts(parsed_agent1, patient_notes)
            responses_agent2 = agent2.generate(prompts=prompts_agent2, batch_size=self.batch_size)
            tokens_agent2 = [agent2.compute_tokens(resp) for resp in responses_agent2]
            raw_agent2 = RegEvaluator.parse_deepseek(responses_agent2)
            parsed_agent2 = RegEvaluator.extract_answer_multiAgent(raw_agent2)

            # --- Agent 3 Processing ---
            prompts_agent3 = self.direct_agent3_prompts(parsed_agent1, parsed_agent2, questions)
            responses_agent3 = agent3.generate(prompts=prompts_agent3, batch_size=self.batch_size)
            tokens_agent3 = [agent3.compute_tokens(resp) for resp in responses_agent3]
            raw_agent3 = RegEvaluator.parse_deepseek(responses_agent3)
            
            # Extract the final answer from Agent 3's responses using the evaluator.
            final_answers = self.evaluator.extract_answer(raw_agent3, calid=calids)
            correctness = self.evaluator.check_correctness(final_answers, ground_truths, calids, upper_limits, lower_limits)
            
            # Compute total token usage per example.
            total_tokens = [t1 + t2 + t3 for t1, t2, t3 in zip(tokens_agent1, tokens_agent2, tokens_agent3)]
            
            # Prepare dictionaries to hold responses and token counts for this group.
            responses_dict = {
                "Agent 1": raw_agent1,
                "Agent 2": raw_agent2,
                "Agent 3": raw_agent3
            }
            tokens_dict = {
                "Agent 1": tokens_agent1,
                "Agent 2": tokens_agent2,
                "Agent 3": tokens_agent3,
                "Total": total_tokens
            }
            
            # Write results to file for this group.
            result_path = self.write_to_file(group_index, responses_dict, tokens_dict, final_answers, correctness, agent1.model_name)
            
            # Compute and save overall accuracy for this group.
            self.evaluator.compute_overall_accuracy(input_file_path=result_path, output_dir_path='json_output')
    
    def write_to_file(self, group_index: int, responses: dict, num_tokens: dict,
                      final_answers: list, correctness: list, modelname: str, filepath: str = 'json_output') -> str:
        """
        Write evaluation results for a specific LLM group to a JSONL file.
        The output JSON includes each agent's response, the final answer,
        correctness, and token usage details.

        Args:
            group_index: Index of the LLM group being evaluated.
            responses: Dictionary containing responses for each agent.
            num_tokens: Dictionary containing token counts for each agent and total.
            final_answers: List of final answers extracted from Agent 3.
            correctness: List of correctness evaluation results.
            filepath: Directory to save the output file.

        Returns:
            str: Path to the created output file.
        """
        os.makedirs(filepath, exist_ok=True)
        output_file = os.path.join(filepath, f"ThreeAgent_group_{group_index}_{modelname}_{self.prompt_style}.jsonl")
        
        records = []
        for i in range(len(self.df)):
            row = self.df.iloc[i]
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
                "Agent 1 Response": responses["Agent 1"][i],
                "Agent 2 Response": responses["Agent 2"][i],
                "Agent 3 Response": responses["Agent 3"][i],
                "LLM Answer": final_answers[i],
                "Result": "Correct" if int(correctness[i]) == 1 else "Incorrect",
                "Agent 1 Tokens": num_tokens["Agent 1"][i],
                "Agent 2 Tokens": num_tokens["Agent 2"][i],
                "Agent 3 Tokens": num_tokens["Agent 3"][i],
                "num_tokens": num_tokens["Total"][i]
            }
            records.append(record)
        
        with open(output_file, 'w') as f:
            for record in records:
                f.write(json.dumps(record, default=str) + '\n')
        
        print(f"Results written to {output_file}")
        return output_file
