from method import Plain, CodeGenerator

import argparse
from model import APIModel
from evaluator import RegEvaluator, LLM_Evaluator
from method import Plain

prompt_styles = ["modular", "modular_cot"]
llm_evaluator = LLM_Evaluator()
reg_evaluator = RegEvaluator()
model = APIModel('OpenAI/gpt-4o-mini')

for prompt_style in prompt_styles:
    
    method = Plain(prompt_style, [model], [reg_evaluator, llm_evaluator])
    raw_json   = method.generate_raw(test=True, raw_json_dir="raw_output/")
    eval_json  = method.evaluate(raw_json,  eval_json_dir="eval_output/")
    reg_evaluator.compute_overall_accuracy_new(eval_json, "stats/")

