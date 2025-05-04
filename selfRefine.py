from model import APIModel
from evaluator import RegEvaluator, LLM_Evaluator
from method import SelfRefine

prompt_styles = ["cot"]
deepseek = APIModel('DeepSeek/deepseek-chat')
llm_evaluator = LLM_Evaluator(deepseek)
for prompt_style in prompt_styles:
    model_name = 'OpenAI/gpt-4o-mini'
    evaluator = RegEvaluator()
    model1 = APIModel(model_name)
    method = SelfRefine(refine_times=3,allow_early_stop=True,prompt_style=prompt_style,llms=[model1],evaluators=[evaluator,llm_evaluator])
    raw = method.generate_raw(test=True)
    eval_json = method.evaluate(raw_json_file=raw)
    evaluator.compute_overall_accuracy_new(eval_json, "stats/SelfRefine/")