import dspy
import pandas as pd
import re
import datetime
import os

from typing import List, Dict, Any
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, BootstrapFinetune
from pydantic import create_model

# when using MIPRO or BootstrapFewShotWithRandomSearch, we need to configure the LM globally or it gives us a 'No LM loaded' error
lm = dspy.OpenAI(model="gpt-4o-mini")
dspy.configure(lm=lm)

def create_custom_signature(input_fields: List[str], output_fields: List[str], instructions: str):
    fields = {field: (str, dspy.InputField(default=..., json_schema_extra={"__dspy_field_type": "input"})) for field in input_fields}
    fields.update({field: (str, dspy.OutputField(default=..., json_schema_extra={"__dspy_field_type": "output"})) for field in output_fields})
    
    CustomSignatureModel = create_model('CustomSignatureModel', **fields)
    
    class CustomSignature(dspy.Signature, CustomSignatureModel):
        """
        {instructions}
        """
    
    CustomSignature.__doc__ = CustomSignature.__doc__.format(instructions=instructions)
    
    return CustomSignature

def generate_human_readable_id(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, optimizer: str, instructions: str) -> str:
    # Create a signature-based name
    signature = '_'.join(input_fields + [':'] + output_fields)
    signature_pascal = ''.join(word.capitalize() for word in signature.split('_'))
    
    # Combine relevant information
    model_name = ''.join(word.capitalize() for word in llm_model.split('-'))
    module_name = dspy_module
    optimizer_name = ''.join(word.capitalize() for word in optimizer.replace('bootstrap', 'bs').replace('randomsearch', 'rs').split('_'))
    
    # Get current date
    current_date = datetime.date.today().strftime("%Y%m%d")
    
    # Create a human-readable ID with date
    unique_id = f"{signature_pascal}-{model_name}_{module_name}_{optimizer_name}-{current_date}"
    
    return unique_id

def compile_program(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, example_data: List[Dict[Any, Any]], optimizer: str, instructions: str) -> str:
    # Set up the LLM model
    if llm_model.startswith("gpt-"):
        lm = dspy.OpenAI(model=llm_model)
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

    # Configure DSPy with the LM
    dspy.configure(lm=lm)

    # Verify that the LM is configured
    assert dspy.settings.lm is not None, "Failed to configure LM"

    # Set up the teacher model
    if teacher_model.startswith("gpt-"):
        teacher_lm = dspy.OpenAI(model=teacher_model)
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model}")

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions)

    # Create the DSPy module
    if dspy_module == "Predict":
        class CustomPredictModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict(CustomSignature)
            
            def forward(self, **kwargs):
                result = self.predictor(**kwargs)
                return dspy.Prediction(**{field: getattr(result, field) for field in output_fields})
        
        module = CustomPredictModule()
    elif dspy_module == "ChainOfThought":
        class CustomChainOfThoughtModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.cot = dspy.ChainOfThought(CustomSignature)
            
            def forward(self, **kwargs):
                return self.cot(**kwargs)
        
        module = CustomChainOfThoughtModule()
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

    # Convert DataFrame to list of dictionaries
    example_data_list = example_data.to_dict('records')

    # Create dataset with correct field names and convert 'funny' to string
    dataset = [dspy.Example(**{input_fields[i]: example[input_fields[i]] for i in range(len(input_fields))},
                            **{output_fields[i]: str(example[output_fields[i]]) for i in range(len(output_fields))}).with_inputs(*input_fields)
               for example in example_data_list]

    # Split the dataset
    split_index = int(0.8 * len(dataset))
    trainset, devset = dataset[:split_index], dataset[split_index:]

    # Set up the metric (always using Exact Match as per the updated interface)
    def metric(gold, pred, trace=None):
        return int(all(gold[field] == pred[field] for field in output_fields))

    # Set up the optimizer
    if optimizer == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=metric, teacher_settings=dict(lm=teacher_lm))
    elif optimizer == "BootstrapFewShotWithRandomSearch":
        teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, teacher_settings=dict(lm=teacher_lm))
    # TODO: add the COPRO optimizer once I get it working
    elif optimizer == "MIPRO":
        teleprompter = MIPRO(metric=metric, teacher_settings=dict(lm=teacher_lm), prompt_model=teacher_lm, requires_permission_to_run=False)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    kwargs = dict(num_threads=4, display_progress=True, display_table=0)
    # Compile the program
    if optimizer == "MIPRO":
        num_trials = 10  # Adjust this value as needed
        max_bootstrapped_demos = 5  # Adjust this value as needed
        max_labeled_demos = 5  # Adjust this value as needed

        compiled_program = teleprompter.compile(module, trainset=trainset, num_trials=num_trials,
    max_bootstrapped_demos=max_bootstrapped_demos,
    max_labeled_demos=max_labeled_demos,
    eval_kwargs=kwargs)
    else:
        compiled_program = teleprompter.compile(module, trainset=trainset, valset=devset)

    # Evaluate the compiled program
    evaluate = Evaluate(metric=metric, devset=devset)
    score = evaluate(compiled_program)

    # Generate a human-readable ID for the compiled program
    human_readable_id = generate_human_readable_id(input_fields, output_fields, dspy_module, llm_model, teacher_model, optimizer, instructions)

    # Create 'programs' folder if it doesn't exist
    os.makedirs('programs', exist_ok=True)

    # Save the compiled program
    compiled_program.save(f"programs/{human_readable_id}.json")

    usage_instructions = f"""Program compiled successfully!
Evaluation score: {score}
The compiled program has been saved as 'programs/{human_readable_id}.json'.
You can now use the compiled program as follows:

compiled_program = dspy.{dspy_module}(CustomSignature)
compiled_program.load('programs/{human_readable_id}.json')
result = compiled_program({', '.join(f'{field}=value' for field in input_fields)})
print({', '.join(f'result.{field}' for field in output_fields)})
"""

    # Use the compiled program with the first row of example data
    if len(example_data) > 0:
        first_row = example_data.iloc[0]
        input_data = {field: first_row[field] for field in input_fields}
        result = compiled_program(**input_data)
        final_prompt = dspy.settings.lm.history[-1]['prompt'] if dspy.settings.lm.history else "No prompt history available"
        
        example_output = f"\nExample usage with first row of data:\n"
        example_output += f"Input: {input_data}\n"
        example_output += f"Output: {result}\n"
        usage_instructions += example_output

    return usage_instructions, final_prompt