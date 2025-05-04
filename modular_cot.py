from model import *
from method import *
from evaluator import *
from typing import List, Tuple



def run():
    model_name = 'OpenAI/gpt-4o-mini'
    filepath = 'raw_output/modular_cot'
    model = APIModel(model_name)
    evaluator = RegEvaluator()
    method = Plain('modular_cot', [model], evaluator)
    method.evaluate()
    result_path = method.write_to_file(model_name, filepath)
    method.evaluator.compute_overall_accuracy(result_path, "stats/")

if __name__ == '__main__':
    run()