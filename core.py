import dspy
import pandas as pd
import json
import ast
from pydantic import Field

from typing import List, Dict, Any
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, BootstrapFinetune, COPRO
from pydantic import create_model

# Helper functions
def load_csv(file_path):
    return pd.read_csv(file_path).to_dict('records')

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

def compile_program(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, example_data: List[Dict[Any, Any]], optimizer: str, instructions: str) -> str:
    # Set up the LLM model
    if llm_model.startswith("gpt-"):
        lm = dspy.OpenAI(model=llm_model)
    elif llm_model.startswith("claude-"):
        lm = dspy.Claude(model=llm_model)
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

    # Configure DSPy with the LM
    dspy.configure(lm=lm)

    # Verify that the LM is configured
    assert dspy.settings.lm is not None, "Failed to configure LM"

    # Set up the teacher model
    if teacher_model.startswith("gpt-"):
        teacher_lm = dspy.OpenAI(model=teacher_model)
    elif teacher_model.startswith("claude-"):
        teacher_lm = dspy.Claude(model=teacher_model)
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model}")

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions)

    # Create the DSPy module
    if dspy_module == "Predict":
        class CustomPredictModule(dspy.Module):
            def __init__(self, lm):
                super().__init__()
                self.predictor = dspy.Predict(CustomSignature)
                self.lm = lm
            
            def forward(self, **kwargs):
                with dspy.context(lm=self.lm):
                    result = self.predictor(**kwargs)
                return dspy.Prediction(**{field: getattr(result, field) for field in output_fields})
        
        module = CustomPredictModule(lm)
    elif dspy_module == "ChainOfThought":
        class CustomChainOfThoughtModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.cot = dspy.ChainOfThought(CustomSignature)
            
            def forward(self, **kwargs):
                return self.cot(**kwargs)
        
        module = CustomChainOfThoughtModule()
    elif dspy_module == "MultiChainComparison":
        class CustomMultiChainComparisonModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.mcc = dspy.MultiChainComparison(CustomSignature)
            
            def forward(self, **kwargs):
                return self.mcc(**kwargs)
        
        module = CustomMultiChainComparisonModule()
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

    # Parse the example data into a dataset
    # Print example data and its type
    print("Example Data:")
    print(example_data)
    print("\nType of example_data:", type(example_data))
    print("\nShape of example_data:", example_data.shape if isinstance(example_data, pd.DataFrame) else "N/A")

    # Convert DataFrame to list of dictionaries
    example_data_list = example_data.to_dict('records')

    # Create dataset with correct field names and convert 'funny' to string
    dataset = [dspy.Example(**{input_fields[i]: example[input_fields[i]] for i in range(len(input_fields))},
                            **{output_fields[i]: str(example[output_fields[i]]) for i in range(len(output_fields))}).with_inputs(*input_fields)
               for example in example_data_list]

    print("Dataset:")
    print(dataset)
    print("\nType of dataset:", type(dataset))
    print("\nLength of dataset:", len(dataset))

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
    elif optimizer == "MIPRO":
        teleprompter = MIPRO(metric=metric, teacher_settings=dict(lm=teacher_lm), prompt_model=teacher_lm)
    elif optimizer == "COPRO":
        teleprompter = COPRO(metric=metric, prompt_model=teacher_lm)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Print LM configuration before compilation
    print("LM configuration before compilation:", dspy.settings.lm)

    # Ensure LM is set in the context for compilation and evaluation
    with dspy.context(lm=lm):
        kwargs = dict(num_threads=4, display_progress=True, display_table=0)
        # Compile the program
        if optimizer == "COPRO":
            compiled_program = teleprompter.compile(module, trainset=trainset, eval_kwargs=kwargs)
        elif optimizer in ["MIPRO", "MIPROv2"]:
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


    # Print out different parts of the compiled program
    print("\nCompiled Program Details:")
    if hasattr(compiled_program, 'predictor'):
        print("Predictor attributes:")
        for attr, value in compiled_program.predictor.__dict__.items():
            print(f"  {attr}: {value}")
    elif hasattr(compiled_program, 'cot'):
        print("ChainOfThought attributes:")
        for attr, value in compiled_program.cot.__dict__.items():
            print(f"  {attr}: {value}")
    elif hasattr(compiled_program, 'mcc'):
        print("MultiChainComparison attributes:")
        for attr, value in compiled_program.mcc.__dict__.items():
            print(f"  {attr}: {value}")
    else:
        print("Compiled program structure:", compiled_program.__dict__)

    # Get the optimized instructions or demonstrations
    optimized_content = "Optimized content not available"
    if hasattr(compiled_program, 'predictor'):
        if hasattr(compiled_program.predictor, 'instructions'):
            optimized_content = f"Instructions:\n{compiled_program.predictor.instructions}"
        if hasattr(compiled_program.predictor, 'demos'):
            optimized_content += f"\n\nDemonstrations:\n{compiled_program.predictor.demos}"
    elif hasattr(compiled_program, 'cot'):
        if hasattr(compiled_program.cot, 'instructions'):
            optimized_content = f"Instructions:\n{compiled_program.cot.instructions}"
        if hasattr(compiled_program.cot, 'demos'):
            optimized_content += f"\n\nDemonstrations:\n{compiled_program.cot.demos}"
    elif hasattr(compiled_program, 'mcc'):
        if hasattr(compiled_program.mcc, 'instructions'):
            optimized_content = f"Instructions:\n{compiled_program.mcc.instructions}"
        if hasattr(compiled_program.mcc, 'demos'):
            optimized_content += f"\n\nDemonstrations:\n{compiled_program.mcc.demos}"

    # If we still couldn't find the optimized content, we can try to inspect the compiled_program object
    if optimized_content == "Optimized content not available":
        optimized_content = f"Could not retrieve optimized content. Compiled program structure: {compiled_program.__dict__}"

    usage_instructions = f"""Program compiled successfully!
Evaluation score: {score}
You can now use the compiled program as follows:

compiled_program = dspy.{dspy_module}(CustomSignature)
compiled_program.load('compiled_program.json')
result = compiled_program({', '.join(f'{field}=value' for field in input_fields)})
print({', '.join(f'result.{field}' for field in output_fields)})
"""

    # Use the compiled program with the first row of example data
    if len(example_data) > 0:
        first_row = example_data.iloc[0]
        input_data = {field: first_row[field] for field in input_fields}
        with dspy.context(lm=lm):
            result = compiled_program(**input_data)
            print('histor', dspy.settings.lm.history)
            final_prompt = lm.history[-1]['prompt'] if lm.history else "No prompt history available"
            print("Final prompt:", final_prompt)
        
        example_output = f"\nExample usage with first row of data:\n"
        example_output += f"Input: {input_data}\n"
        example_output += f"Output: {result}\n"
        usage_instructions += example_output

    return usage_instructions, optimized_content, final_prompt