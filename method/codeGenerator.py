from model import LLM
from typing import Union, Tuple, List
import json
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from method import Method

class CodeGenerator(Method):
    """
    A class to generate and execute code based on formulas and extracted values using an LLM model.
    """

    # Commonly available Python standard and preinstalled packages
    COMMON_PACKAGES = [
        'math', 'datetime', 'json', 're', 'os', 'sys',
        'numpy', 'pandas'
    ]

    def __init__(self, llms: List[LLM]):

        super().__init__(llms=llms)
        self.llms = llms
        self.input_tokens_used = 0
        self.output_tokens_used = 0

    def generate_code(
        self,
        formulas: List[Union[str, dict]],
        extracted_values: List[Union[str, dict]]
    ) -> List[str]:
        """
        Generates and executes code for each formula and extracted values.

        Args:
            formulas: A list of formulas (string or dict) to convert into code.
            extracted_values: A list of values (string or dict) associated with each formula.

        Returns:
            A list of results obtained by executing the generated code.
        """
        # Parse JSON strings if necessary
        parsed_formulas = []
        parsed_values = []
        for f, v in zip(formulas, extracted_values):
            if isinstance(f, str):
                try:
                    f = json.loads(f)
                except json.JSONDecodeError:
                    pass
            if isinstance(v, str):
                try:
                    v = json.loads(v)
                except json.JSONDecodeError:
                    pass
            parsed_formulas.append(f)
            parsed_values.append(v)

        # Generate prompts
        prompts = self._gen_code_prompt(parsed_formulas, parsed_values)

        # Call the model to generate code
        responses, input_tokens, output_tokens = self.model.generate(prompts)
        # Update input and output tokens used
        self.input_tokens_used += np.sum(input_tokens)
        self.output_tokens_used += np.sum(output_tokens)

        # Extract code from responses and execute
        return self._run_responses(responses)

    def _gen_code_prompt(
        self,
        formulas: List[Union[str, dict]],
        extracted_values: List[Union[str, dict]]
    ) -> List[Tuple[str, str]]:
        """
        Creates system and user messages for the LLM to generate code.
        """
        prompts: List[Tuple[str, str]] = []
        system_msg = (
            "You are a Python code generator. "
            "Generate executable Python code that computes the result of a given formula using provided values. "
            f"Only use commonly available packages: {', '.join(self.COMMON_PACKAGES)}. "
            "Store the final output in a variable named 'result'."
        )

        for formula, values in zip(formulas, extracted_values):
            if formula == values:
                user_msg = (
                    f"Formula and Values: {json.dumps(formula)}\n"
                    "# Write Python code below to compute 'result'"
                )
            else:
                user_msg = (
                    f"Formula: {json.dumps(formula)}\n"
                    f"Values: {json.dumps(values)}\n"
                    "# Write Python code below to compute 'result'"
                )
            prompts.append((system_msg, user_msg))

        return prompts

    async def _execute_code(self, code: str) -> str:
        """
        Asynchronously executes the generated code and returns the result.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.execute_code, code)

    def execute_code(self, code: str) -> str:
        """
        Executes the generated Python code in a sandboxed environment and returns the result.

        Args:
            code: The Python code to execute. Assumes that the code defines a variable 'result'.

        Returns:
            The string representation of 'result' or error message if execution fails.
        """
        local_vars = {}
        try:
            exec(code, {}, local_vars)
            result = local_vars.get('result', None)
            return str(result)
        except Exception as e:
            # Return the error for debugging
            return f"Execution error: {e}"

    def _run_responses(self, responses, max_workers: int = 4) -> List[str]:
        """
        Executes the generated code in parallel using ThreadPoolExecutor.
        Args:
            responses: List of generated code strings to execute
            max_workers: Number of threads to use for parallel execution
        Returns:
            List of results from executing the code.
        """
        # Initialize results list with None
        results = [None] * len(responses)
        # Map each submitted future to its original index
        futures = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, resp in enumerate(responses):
                futures[executor.submit(self.execute_code, resp.strip())] = idx

            # as_completed gives futures as they finish; tqdm tracks overall progress
            for future in tqdm(as_completed(futures),
                            total=len(futures),
                            desc="Executing code"):
                idx = futures[future]
                results[idx] = future.result()

        return results