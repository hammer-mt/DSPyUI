import dspy
import pandas as pd
import re
import datetime
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from typing import List, Dict, Any
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, MIPROv2, COPRO, BootstrapFinetune
from pydantic import create_model

# List of supported Groq models
SUPPORTED_GROQ_MODELS = [
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it"
]

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

def create_dspy_module(dspy_module: str, CustomSignature: type) -> dspy.Module:
    if dspy_module == "Predict":
        class CustomPredictModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict(CustomSignature)
            
            def forward(self, **kwargs):
                result = self.predictor(**kwargs)
                return dspy.Prediction(**{field: getattr(result, field) for field in CustomSignature.__annotations__ if field not in CustomSignature.__fields_set__})
        
        return CustomPredictModule()
    elif dspy_module == "ChainOfThought":
        class CustomChainOfThoughtModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.cot = dspy.ChainOfThought(CustomSignature)
            
            def forward(self, **kwargs):
                return self.cot(**kwargs)
        
        return CustomChainOfThoughtModule()
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

def compile_program(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, example_data: List[Dict[Any, Any]], optimizer: str, instructions: str, metric_type: str, judge_prompt_id=None) -> str:
    # Set up the LLM model
    if llm_model.startswith("gpt-"):
        lm = dspy.OpenAI(model=llm_model)
    elif llm_model.startswith("claude-"):
        lm = dspy.Claude(model=llm_model)
    elif llm_model in SUPPORTED_GROQ_MODELS:
        lm = dspy.GROQ(model=llm_model, api_key=os.environ.get("GROQ_API_KEY"))
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
    elif teacher_model in SUPPORTED_GROQ_MODELS:
        teacher_lm = dspy.GROQ(model=teacher_model, api_key=os.environ.get("GROQ_API_KEY"))
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model}")

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions)

    # Create the DSPy module using the new function
    module = create_dspy_module(dspy_module, CustomSignature)

    # Convert DataFrame to list of dictionaries
    example_data_list = example_data.to_dict('records')

    # Check if there are at least two examples
    if len(example_data_list) < 2:
        raise ValueError("At least two examples are required for compilation.")

    # Create dataset with correct field names and convert 'funny' to string
    dataset = [dspy.Example(**{input_fields[i]: example[input_fields[i]] for i in range(len(input_fields))},
                            **{output_fields[i]: str(example[output_fields[i]]) for i in range(len(output_fields))}).with_inputs(*input_fields)
               for example in example_data_list]

    # Split the dataset
    split_index = int(0.8 * len(dataset))
    trainset, devset = dataset[:split_index], dataset[split_index:]

    # Set up the evaluation metric
    if metric_type == "Exact Match":
        def metric(gold, pred, trace=None):
            return int(all(gold[field] == pred[field] for field in output_fields))
    elif metric_type == "Cosine Similarity":
        # Initialize the OpenAI client
        client = OpenAI()

        def get_embedding(text):
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding

        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        def metric(gold, pred, trace=None):
            gold_vector = np.concatenate([get_embedding(str(gold[field])) for field in output_fields])
            pred_vector = np.concatenate([get_embedding(str(pred[field])) for field in output_fields])
            
            similarity = cosine_similarity(gold_vector, pred_vector)

            return similarity
    elif metric_type == "LLM-as-a-Judge":
        if judge_prompt_id is None:
            raise ValueError("Judge prompt ID is required for LLM-as-a-Judge metric")
        
        # Load the compiled judge program
        judge_program_path = f"programs/{judge_prompt_id}.json"
        if not os.path.exists(judge_program_path):
            raise ValueError(f"Judge program not found: {judge_program_path}")
        
        # Load the judge prompt details
        with open(f"prompts/{judge_prompt_id}.json", 'r') as f:
            judge_prompt_details = json.load(f)
        
        judge_input_fields = judge_prompt_details.get('input_fields', [])
        judge_output_fields = judge_prompt_details.get('output_fields', [])
        judge_module = judge_prompt_details.get('dspy_module', 'Predict')  # Default to 'Predict' if not specified

        print("Judge Prompt Details Loaded:")
        print(judge_prompt_details)
        
        print("Judge Input Fields:")
        print(judge_input_fields)
        print("Judge Output Fields:")
        print(judge_output_fields)
        
        # Recreate the custom signature for the judge program
        JudgeSignature = create_custom_signature(judge_input_fields, judge_output_fields, judge_prompt_details.get('instructions', ''))
        
        # Load the compiled judge program
        judge_program = create_dspy_module(judge_module, JudgeSignature)
        judge_program.load(judge_program_path)
        
        def metric(gold, pred, trace=None):
            try:
                # Prepare input for the judge program based on judge_input_fields
                judge_input = {}
                for field in judge_input_fields:
                    if field in gold:
                        judge_input[field] = gold[field]
                    elif field in pred:
                        judge_input[field] = pred[field]
                    else:
                        print(f"Warning: Required judge input field '{field}' not found in gold or pred")
                        judge_input[field] = ""  # or some default value
                
                print("Judge Input Prepared:")
                print(judge_input)
                
                # Run the judge program
                result = judge_program(**judge_input)
                
                print("Judge Program Result:")
                print(result)
                
                # Extract the score from the judge output
                if len(judge_output_fields) == 1:
                    score_field = judge_output_fields[0]
                    if hasattr(result, score_field):
                        return float(getattr(result, score_field))
                    else:
                        print(f"Warning: Judge program did not return expected field '{score_field}'")
                        return 0.0
                else:
                    print("Warning: Multiple or no output fields in judge program. Using first field as score.")
                    return float(getattr(result, judge_output_fields[0], 0.0))
            except Exception as e:
                print(f"Error in metric function: {str(e)}")
                return 0.0  # Return a default score in case of error
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    # Set up the optimizer
    if optimizer == "None":
        # Skip compilation, use the module as-is
        compiled_program = module
    elif optimizer == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=metric, teacher_settings=dict(lm=teacher_lm))
        compiled_program = teleprompter.compile(module, trainset=trainset)
    elif optimizer == "BootstrapFewShotWithRandomSearch":
        teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, teacher_settings=dict(lm=teacher_lm), num_threads=1)
        compiled_program = teleprompter.compile(module, trainset=trainset, valset=devset)
    elif optimizer == "COPRO":
        teleprompter = COPRO(metric=metric, teacher_settings=dict(lm=teacher_lm))
        compiled_program = teleprompter.compile(module, trainset=trainset, valset=devset)
    elif optimizer == "MIPRO":
        teleprompter = MIPRO(metric=metric, teacher_settings=dict(lm=teacher_lm), prompt_model=teacher_lm, task_model=lm)
        num_trials = 10  # Adjust this value as needed
        max_bootstrapped_demos = 5  # Adjust this value as needed
        max_labeled_demos = 5  # Adjust this value as needed
        compiled_program = teleprompter.compile(module, trainset=trainset, num_trials=num_trials,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            eval_kwargs=kwargs, requires_permission_to_run=False)
    elif optimizer == "MIPROv2":
        teleprompter = MIPROv2(metric=metric, teacher_settings=dict(lm=teacher_lm))
        num_batches = 30
        max_bootstrapped_demos = 5
        max_labeled_demos = 2
        compiled_program = teleprompter.compile(
            module,
            trainset=trainset,
            valset=devset,
            num_batches=num_batches,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            eval_kwargs=kwargs,
            requires_permission_to_run=False
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Use a single thread for evaluation
    kwargs = dict(num_threads=1, display_progress=True, display_table=1)

    # Evaluate the compiled program
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1)
    score = evaluate(compiled_program)

    print("Evaluation Score:")
    print(score)

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