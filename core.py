
import dspy
import pandas as pd

from typing import List, Dict, Any
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, BootstrapFinetune, COPRO

# Helper functions
def load_csv(file_path):
    return pd.read_csv(file_path).to_dict('records')

def compile_program(input_fields: List[str], output_fields: List[str], llm_model: str, dspy_module: str, example_data: List[Dict[Any, Any]], 
                    metric_type: str, optimizer: str) -> str:
    # Set up the LLM model
    if llm_model.startswith("gpt-"):
        lm = dspy.OpenAI(model=llm_model)
    elif llm_model.startswith("claude-"):
        lm = dspy.Claude(model=llm_model)
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

    dspy.configure(lm=lm)

    # Define the signature
    class CustomSignature(dspy.Signature):
        def __init__(self):
            super().__init__()
            self.input_fields = [dspy.InputField(name=field) for field in input_fields]
            self.output_fields = [dspy.OutputField(name=field) for field in output_fields]

    # Create the DSPy module
    if dspy_module == "Predict":
        module = dspy.Predict(CustomSignature)
    elif dspy_module == "ChainOfThought":
        module = dspy.ChainOfThought(CustomSignature)
    elif dspy_module == "ReAct":
        module = dspy.ReAct(CustomSignature)
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

    # Prepare the dataset
    dataset = [dspy.Example(**example).with_inputs(*input_fields) for example in example_data]
    
    # Split the dataset
    split_index = int(0.8 * len(dataset))
    trainset, devset = dataset[:split_index], dataset[split_index:]

    # Set up the metric
    if metric_type == "Exact Match":
        def metric(gold, pred, trace=None):
            return int(all(gold[field] == pred[field] for field in output_fields))
    else:
        raise ValueError(f"Unsupported metric type: {metric_type}")

    # Set up the optimizer
    if optimizer == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=metric)
    elif optimizer == "BootstrapFewShotWithRandomSearch":
        teleprompter = BootstrapFewShotWithRandomSearch(metric=metric)
    elif optimizer == "MIPRO":
        teleprompter = MIPRO(metric=metric)
    elif optimizer == "BootstrapFinetune":
        teleprompter = BootstrapFinetune(metric=metric)
    elif optimizer == "COPRO":
        teleprompter = COPRO(metric=metric)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Compile the program
    compiled_program = teleprompter.compile(module, trainset=trainset, valset=devset)

    # Evaluate the compiled program
    evaluate = Evaluate(metric=metric, devset=devset)
    score = evaluate(compiled_program)

    return f"""Program compiled successfully!
Evaluation score: {score}
You can now use the compiled program as follows:

compiled_program = dspy.ChainOfThought(CustomSignature)
compiled_program.load('compiled_program.json')
result = compiled_program({', '.join(f'{field}=value' for field in input_fields)})
print({', '.join(f'result.{field}' for field in output_fields)})
"""