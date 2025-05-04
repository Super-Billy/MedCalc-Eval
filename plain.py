import argparse
from model import APIModel
from evaluator import RegEvaluator, LLM_Evaluator
from method import Plain

prompt_styles = ["oneshot", "modular", "modular_cot"]
llm_evaluator = LLM_Evaluator(APIModel('OpenAI/gpt-4.1-mini', tpm_limit=150000000, rpm_limit=30000))
reg_evaluator = RegEvaluator()
for prompt_style in prompt_styles:
    model = APIModel('OpenAI/gpt-4o-mini', tpm_limit=150000000, rpm_limit=30000)
    method = Plain(prompt_style, [model], [reg_evaluator, llm_evaluator])
    raw_json   = method.generate_raw(test=False, raw_json_dir="raw_output/")
    eval_json  = method.evaluate(raw_json,  eval_json_dir="eval_output/")
    reg_evaluator.compute_overall_accuracy_new(eval_json, "stats/")

