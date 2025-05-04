from model import APIModel
from evaluator import RegEvaluator, LLM_Evaluator
from method import TwoAgent

prompt_styles = ["cot"]
deepseek = APIModel('DeepSeek/deepseek-chat')
llm_evaluator = LLM_Evaluator(deepseek)
for prompt_style in prompt_styles:
    model_name = 'OpenAI/gpt-4o-mini'
    evaluator = RegEvaluator()
    model1 = APIModel(model_name)
    model2 = APIModel(model_name)
    method = TwoAgent(prompt_style, [[model1, model2]], [evaluator,llm_evaluator])
    raw = method.generate_raw(test=True,use_rag=False)
    # eval_json = method.evaluate(raw_json_file=raw)
    # evaluator.compute_overall_accuracy_new(eval_json, "stats/twoAgent/")