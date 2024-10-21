import dspy
import pandas as pd
import re
import datetime
import os
import json
import numpy as np
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

SUPPORTED_GOOGLE_MODELS = [
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

# when using MIPRO or BootstrapFewShotWithRandomSearch, we need to configure the LM globally or it gives us a 'No LM loaded' error
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

def create_custom_signature(input_fields: List[str], output_fields: List[str], instructions: str, input_descs: List[str], output_descs: List[str]):
    fields = {}
    for i, field in enumerate(input_fields):
        if i < len(input_descs) and input_descs[i]:
            fields[field] = (str, dspy.InputField(default=..., desc=input_descs[i], json_schema_extra={"__dspy_field_type": "input"}))
        else:
            fields[field] = (str, dspy.InputField(default=..., json_schema_extra={"__dspy_field_type": "input"}))
    
    for i, field in enumerate(output_fields):
        if i < len(output_descs) and output_descs[i]:
            fields[field] = (str, dspy.OutputField(default=..., desc=output_descs[i], json_schema_extra={"__dspy_field_type": "output"}))
        else:
            fields[field] = (str, dspy.OutputField(default=..., json_schema_extra={"__dspy_field_type": "output"}))
    
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

def create_dspy_module(dspy_module: str, CustomSignature: type, hint: str = None) -> dspy.Module:
    if dspy_module == "Predict":
        class CustomPredictModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.predictor = dspy.Predict(CustomSignature)
            
            def forward(self, **kwargs):
                result = self.predictor(**kwargs)
                return result
        
        return CustomPredictModule()
    elif dspy_module == "ChainOfThought":
        class CustomChainOfThoughtModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.cot = dspy.ChainOfThought(CustomSignature)
            
            def forward(self, **kwargs):
                return self.cot(**kwargs)
        
        return CustomChainOfThoughtModule()
    elif dspy_module == "ChainOfThoughtWithHint":
        class CustomChainOfThoughtWithHintModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.cot_with_hint = dspy.ChainOfThought(CustomSignature)
                self.hint = hint
            
            def forward(self, **kwargs):
                # Inject the hint into the kwargs
                kwargs['hint'] = self.hint
                return self.cot_with_hint(**kwargs)
        
        return CustomChainOfThoughtWithHintModule()
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

def compile_program(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, example_data: List[Dict[Any, Any]], optimizer: str, instructions: str, metric_type: str, judge_prompt_id=None, input_descs: List[str] = None, output_descs: List[str] = None, hint: str = None) -> str:
    # Set up the LLM model
    if llm_model.startswith("gpt-"):
        lm = dspy.LM(f'openai/{llm_model}')
    elif llm_model.startswith("claude-"):
        lm = dspy.LM(f'anthropic/{llm_model}')
    elif llm_model in SUPPORTED_GROQ_MODELS:
        lm = dspy.LM(f'groq/{llm_model}', api_key=os.environ.get("GROQ_API_KEY"))
    elif llm_model in SUPPORTED_GOOGLE_MODELS:
        lm = dspy.LM(f'google/{llm_model}', api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unsupported LLM model: {llm_model}")

    # Configure DSPy with the LM
    dspy.configure(lm=lm)

    # Verify that the LM is configured
    assert dspy.settings.lm is not None, "Failed to configure LM"

    # Set up the teacher model
    if teacher_model.startswith("gpt-"):
        teacher_lm = dspy.LM(f'openai/{teacher_model}')
    elif teacher_model.startswith("claude-"):
        teacher_lm = dspy.LM(f'anthropic/{teacher_model}')
    elif teacher_model in SUPPORTED_GROQ_MODELS:
        teacher_lm = dspy.LM(f'groq/{teacher_model}', api_key=os.environ.get("GROQ_API_KEY"))
    elif teacher_model in SUPPORTED_GOOGLE_MODELS:
        teacher_lm = dspy.LM(f'google/{teacher_model}', api_key=os.environ.get("GOOGLE_API_KEY"))
    else:
        raise ValueError(f"Unsupported teacher model: {teacher_model}")

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions, input_descs or [], output_descs or [])

    # Create the DSPy module using the new function
    module = create_dspy_module(dspy_module, CustomSignature, hint)

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
            print("Gold:", gold)
            print("Pred:", pred)
            print("Pred type:", type(pred))
            print("Pred attributes:", dir(pred))
            
            if isinstance(pred, dspy.Prediction):
                print("Prediction fields:", pred.__dict__)
            
            # Check if pred is empty or None
            if not pred or (isinstance(pred, dspy.Prediction) and not pred.__dict__):
                print("Warning: Prediction is empty or None")
                return 0
            
            try:
                return int(all(gold[field] == getattr(pred, field) for field in output_fields))
            except AttributeError as e:
                print(f"AttributeError: {e}")
                return 0
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
        
        example2_id = "JokeTopic:Funny-Gpt4oMini_ChainOfThought_Bootstrapfewshotwithrandomsearch-20241003.json"
        
        # Load the judge prompt details
        if judge_prompt_id == example2_id:
            judge_prompt_path = f"example_data/{judge_prompt_id}"
        else:
            judge_prompt_path = f"prompts/{judge_prompt_id}.json"
        
        if not os.path.exists(judge_prompt_path):
            raise ValueError(f"Judge prompt not found: {judge_prompt_path}")
        
        with open(judge_prompt_path, 'r') as f:
            judge_prompt_details = json.load(f)

        print("Judge Prompt Path:", judge_prompt_path)
        print("Judge Prompt Details:", judge_prompt_details)
        
        judge_input_fields = judge_prompt_details.get('input_fields', [])
        judge_input_descs = judge_prompt_details.get('input_descs', [])
        judge_output_fields = judge_prompt_details.get('output_fields', [])
        judge_output_descs = judge_prompt_details.get('output_descs', [])
        judge_module = judge_prompt_details.get('dspy_module', 'Predict')
        judge_instructions = judge_prompt_details.get('instructions', '')
        judge_human_readable_id = judge_prompt_details.get('human_readable_id')

        print("Judge Prompt Details:")
        print(json.dumps(judge_prompt_details, indent=2))
        
        # Create the custom signature for the judge program
        JudgeSignature = create_custom_signature(judge_input_fields, judge_output_fields, judge_instructions, judge_input_descs, judge_output_descs)
        
        print("\nJudge Signature:")
        print(JudgeSignature)
        
        # Create the judge program
        judge_program = create_dspy_module(judge_module, JudgeSignature)
        
        print("\nJudge Program:")
        print(judge_program)
        
        # Load the compiled judge program
        if judge_prompt_id == example2_id:
            judge_program_path = f"example_data/{judge_human_readable_id}-program.json"
        else:
            judge_program_path = f"programs/{judge_human_readable_id}.json"
        
        if not os.path.exists(judge_program_path):
            raise ValueError(f"Compiled judge program not found: {judge_program_path}")
        
        with open(judge_program_path, 'r') as f:
            judge_program_content = json.load(f)
        
        print("\nCompiled Judge Program Content:")
        print(json.dumps(judge_program_content, indent=2))
        
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
                
                print("Judge Input:")
                print(json.dumps(judge_input, indent=2))
                
                # Run the judge program
                result = judge_program(**judge_input)
                
                print("Judge Program Result:")
                print(result)
                print("Result type:", type(result))
                print("Result attributes:", dir(result))
                if hasattr(result, 'toDict'):
                    print("Result as dict:", result.toDict())
                
                # Extract the score from the judge output
                if len(judge_output_fields) == 1:
                    score_field = judge_output_fields[0]
                    if hasattr(result, score_field):
                        score = getattr(result, score_field)
                        print(f"Score: {score}")
                        return float(score)
                    else:
                        # If the score field is not directly accessible, try to access it from the result dictionary
                        result_dict = result.toDict() if hasattr(result, 'toDict') else {}
                        if score_field in result_dict:
                            score = result_dict[score_field]
                            print(f"Score: {score}")
                            return float(score)
                        else:
                            print(f"Error: Judge program did not return expected field '{score_field}'")
                            print(f"Available fields: {result_dict.keys() if result_dict else dir(result)}")
                            return 0.0
                else:
                    print(f"Error: Expected 1 output field, got {len(judge_output_fields)}")
                    print(f"Output fields: {judge_output_fields}")
                    return 0.0
            except Exception as e:
                print(f"Error in metric function: {str(e)}")
                return 0.0  # Return a default score in case of error
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")
    
    # Use a single thread for evaluation
    kwargs = dict(num_threads=1, display_progress=True, display_table=1)

    # Evaluate the module to establish a baseline
    baseline_evaluate = Evaluate(metric=metric, devset=devset, num_threads=1)
    baseline_score = baseline_evaluate(module)

    # Set up the optimizer
    if optimizer == "BootstrapFewShot":
        teleprompter = BootstrapFewShot(metric=metric, teacher_settings=dict(lm=teacher_lm))
        compiled_program = teleprompter.compile(module, trainset=trainset)
    elif optimizer == "BootstrapFewShotWithRandomSearch":
        teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, teacher_settings=dict(lm=teacher_lm), num_threads=1)
        compiled_program = teleprompter.compile(module, trainset=trainset, valset=devset)
    elif optimizer == "COPRO":
        teleprompter = COPRO(metric=metric, teacher_settings=dict(lm=teacher_lm))
        compiled_program = teleprompter.compile(module, trainset=trainset, eval_kwargs=kwargs)
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
        teleprompter = MIPROv2(metric=metric, prompt_model=lm, task_model=teacher_lm, num_candidates=10, init_temperature=1.0)

        num_batches = 30
        max_bootstrapped_demos = 8
        max_labeled_demos = 16
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

    # Evaluate the compiled program
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1)
    score = evaluate(compiled_program)

    print("Evaluation Score:")
    print(score)

    # Generate a human-readable ID for the compiled program
    human_readable_id = generate_human_readable_id(input_fields, output_fields, dspy_module, llm_model, teacher_model, optimizer, instructions)

    # Create datasets folder if it doesn't exist
    os.makedirs('datasets', exist_ok=True)

    # Save the dataframe to the datasets folder
    dataset_path = os.path.join('datasets', f"{human_readable_id}.csv")
    example_data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")


    # Create 'programs' folder if it doesn't exist
    os.makedirs('programs', exist_ok=True)

    # Save the compiled program
    compiled_program.save(f"programs/{human_readable_id}.json")
    print(f"Compiled program saved to programs/{human_readable_id}.json")

    usage_instructions = f"""Program compiled successfully!
Evaluation score: {score}
Baseline score: {baseline_score}
The compiled program has been saved as 'programs/{human_readable_id}.json'.
You can now use the compiled program as follows:

compiled_program = dspy.{dspy_module}(CustomSignature)
compiled_program.load('programs/{human_readable_id}.json')
result = compiled_program({', '.join(f'{field}=value' for field in input_fields)})
print({', '.join(f'result.{field}' for field in output_fields)})
"""

    # Update the usage instructions to include the hint if applicable
    if dspy_module == "ChainOfThoughtWithHint":
        usage_instructions += f"\nHint: {hint}\n"

    # Use the compiled program with the first row of example data
    if len(example_data) > 0:
        first_row = example_data.iloc[0]
        input_data = {field: first_row[field] for field in input_fields}
        result = compiled_program(**input_data)
        messages = dspy.settings.lm.history[-1]['messages']
        final_prompt = ""
        for msg in messages:
            final_prompt += f"{msg['content']}\n"

        example_output = f"\nExample usage with first row of data:\n"
        example_output += f"Input: {input_data}\n"
        example_output += f"Output: {result}\n"
        usage_instructions += example_output

    return usage_instructions, final_prompt

# Function to list prompts
def list_prompts(signature_filter=None, output_filter=None):
    
    if not os.path.exists('prompts'):
        print("Prompts directory does not exist")
        return []
    
    files = os.listdir('prompts')
    if not files:
        print("No prompt files found in the prompts directory")
        return []
    
    prompt_details = []
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join('prompts', file), 'r') as f:
                data = json.load(f)
                prompt_id = file
                signature = f"{', '.join(data['input_fields'])} -> {', '.join(data['output_fields'])}"
                
                input_signature = f"{', '.join(data['input_fields'])}"
                
                eval_score = data.get('evaluation_score', 'N/A')
                # Exclude example data
                details = {k: v for k, v in data.items() if k != 'example_data'}
                
                # Check if signature_filter is provided and matches
                if signature_filter and signature_filter.lower() not in signature.lower():
                    print(f"Skipping file {file} due to signature mismatch")
                    continue

                # Check if output_filter is provided and matches
                if output_filter:
                    if not all(filter_item.lower() in input_signature.lower() for filter_item in output_filter):
                        continue
                
                prompt_details.append({
                    "ID": prompt_id,
                    "Signature": signature,
                    "Eval Score": eval_score,
                    "Details": json.dumps(details, indent=4)  # Add full details as a JSON string
                })
    
    print(f"Found {len(prompt_details)} saved prompts")
    return prompt_details  # Return the list of prompts as dictionaries

def load_example_csv(example_name):
    csv_path = f"example_data/{example_name}.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return None


def export_to_csv(data):
    df = pd.DataFrame(data)
    filename = "exported_data.csv"
    df.to_csv(filename, index=False)
    return filename


# function to take a program from the program folder and run it on a row from the dataset
def generate_program_response(human_readable_id, row_data):
    # Load the program details
    program_path = f"programs/{human_readable_id}.json"
    prompt_path = f"prompts/{human_readable_id}.json"

    print("program_path:", program_path)
    
    if not os.path.exists(program_path):
        raise ValueError(f"Compiled program not found: {program_path}")

    with open(prompt_path, 'r') as f:
        program_details = json.load(f)
    
    # Extract necessary information from program details
    input_fields = program_details.get('input_fields', [])
    input_descs = program_details.get('input_descs', [])    
    output_fields = program_details.get('output_fields', [])
    output_descs = program_details.get('output_descs', [])
    dspy_module = program_details.get('dspy_module', 'Predict')
    instructions = program_details.get('instructions', '')

    print("input_fields:", input_fields)
    print("output_fields:", output_fields)
    print("instructions:", instructions)
    print("input_descs:", input_descs)
    print("output_descs:", output_descs)
    print("dspy_module:", dspy_module)

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions, input_descs, output_descs)
    print("CustomSignature:", CustomSignature)
    compiled_program = create_dspy_module(dspy_module, CustomSignature)
    print("compiled_program:", compiled_program)
    compiled_program.load(program_path)
    print("compiled_program after load:", compiled_program)


    program_input = {}
    for field in input_fields:
        if field in row_data:
            program_input[field] = row_data[field]
        else:
            print(f"Warning: Required input field '{field}' not found in row_data")
            program_input[field] = ""  # or some default value
    
    # Run the program

    try:
        result = compiled_program(**program_input)
        print("result:", result)
    except Exception as e:
        print(f"Error executing program: {str(e)}")
        return f"Error: {str(e)}"

    # Prepare the output
    output = "Input:\n"
    for field in input_fields:
        output += f"{field}: {program_input[field]}\n"

    print("result:", result)

    output += "\nOutput:\n"
    for field in output_fields:
        output += f"{field}: {getattr(result, field)}\n"

    print("output:", output)

    return output
