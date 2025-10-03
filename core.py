import dspy
import pandas as pd
import re
import datetime
import os
import json
import numpy as np
from openai import OpenAI

from typing import List, Dict, Any, Optional, Tuple, Callable, Type, Union
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO, BootstrapFinetune, LabeledFewShot
from pydantic import create_model, BaseModel
import numpy.typing as npt

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

# Model pricing in USD per 1M tokens (input, output)
# Prices as of January 2025
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic models
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Groq models
    "mixtral-8x7b-32768": {"input": 0.27, "output": 0.27},
    "llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "gemma-7b-it": {"input": 0.07, "output": 0.07},
    "gemma2-9b-it": {"input": 0.20, "output": 0.20},

    # Google models
    "gemini-1.5-flash-8b": {"input": 0.04, "output": 0.15},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},

    # Local models have zero cost
    "local": {"input": 0.0, "output": 0.0}
}

# when using MIPRO or BootstrapFewShotWithRandomSearch, we need to configure the LM globally or it gives us a 'No LM loaded' error
lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

def create_custom_signature(
    input_fields: List[str],
    output_fields: List[str],
    instructions: str,
    input_descs: List[str],
    output_descs: List[str]
) -> Any:  # Return Any since we're creating a dynamic type that inherits from dspy.Signature
    fields: Dict[str, Any] = {}
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

    # Create a BaseModel subclass dynamically
    CustomSignatureModel: Type[BaseModel] = create_model('CustomSignatureModel', **fields)

    class CustomSignature(dspy.Signature, CustomSignatureModel):  # type: ignore[misc,valid-type]
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

def create_dspy_module(
    dspy_module: str,
    CustomSignature: Any,  # Dynamic signature type from create_custom_signature
    hint: Optional[str] = None,
    max_iters: int = 3
) -> Any:  # Returns a dspy.Module subclass
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
    elif dspy_module == "ProgramOfThought":
        class CustomProgramOfThoughtModule(dspy.Module):
            def __init__(self):
                super().__init__()
                self.pot = dspy.ProgramOfThought(CustomSignature, max_iters=max_iters)

            def forward(self, **kwargs):
                return self.pot(**kwargs)

        return CustomProgramOfThoughtModule()
    else:
        raise ValueError(f"Unsupported DSPy module: {dspy_module}")

def compile_program(
    input_fields: List[str],
    output_fields: List[str],
    dspy_module: str,
    llm_model: str,
    teacher_model: str,
    example_data: pd.DataFrame,
    optimizer: str,
    instructions: str,
    metric_type: str,
    judge_prompt_id: Optional[str] = None,
    input_descs: Optional[List[str]] = None,
    output_descs: Optional[List[str]] = None,
    hint: Optional[str] = None,
    max_iters: int = 3,
    k: int = 16,
    llm_base_url: Optional[str] = None,
    teacher_base_url: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    # Set up the LLM model
    if llm_model.startswith("local:"):
        # Local LLM with custom endpoint
        model_name = llm_model[6:]  # Remove "local:" prefix
        base_url = llm_base_url or "http://127.0.0.1:1234/v1"
        lm = dspy.LM(f'openai/{model_name}', api_base=base_url)
    elif llm_model.startswith("gpt-"):
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
    if teacher_model.startswith("local:"):
        # Local LLM with custom endpoint
        model_name = teacher_model[6:]  # Remove "local:" prefix
        base_url = teacher_base_url or llm_base_url or "http://127.0.0.1:1234/v1"
        teacher_lm = dspy.LM(f'openai/{model_name}', api_base=base_url)
    elif teacher_model.startswith("gpt-"):
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
    module = create_dspy_module(dspy_module, CustomSignature, hint, max_iters)

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
    elif optimizer == "LabeledFewShot":
        teleprompter = LabeledFewShot(k=k)
        compiled_program = teleprompter.compile(module, trainset=trainset)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer}")

    # Evaluate the compiled program
    evaluate = Evaluate(metric=metric, devset=devset, num_threads=1)
    score = evaluate(compiled_program)

    print("Evaluation Score:")
    print(score)

    # Calculate actual cost from DSPy history
    actual_cost_data = calculate_actual_cost(llm_model, teacher_model)
    print(f"Actual cost: ${actual_cost_data['actual_cost_usd']:.4f}")
    print(f"Total tokens: {actual_cost_data['total_tokens']:,} ({actual_cost_data['input_tokens']:,} input, {actual_cost_data['output_tokens']:,} output)")

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

    return usage_instructions, final_prompt, actual_cost_data

# Function to list prompts
def list_prompts(
    signature_filter: Optional[str] = None,
    output_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    
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
                actual_cost = data.get('actual_cost', 0)
                total_tokens = data.get('total_tokens', 0)

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
                    "Cost": actual_cost,
                    "Tokens": total_tokens,
                    "Details": json.dumps(details, indent=4)  # Add full details as a JSON string
                })
    
    print(f"Found {len(prompt_details)} saved prompts")
    return prompt_details  # Return the list of prompts as dictionaries

def load_example_csv(example_name: str) -> Optional[pd.DataFrame]:
    """Load an example CSV file from the example_data directory."""
    csv_path = f"example_data/{example_name}.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return None


def export_to_csv(data: List[Dict[str, Any]]) -> str:
    """Export data to a CSV file."""
    df = pd.DataFrame(data)
    filename = "exported_data.csv"
    df.to_csv(filename, index=False)
    return filename


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing information for a model.

    Args:
        model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet-20240620')

    Returns:
        Dictionary with 'input' and 'output' prices per 1M tokens in USD
    """
    # Handle local models
    if model.startswith("local:"):
        return MODEL_PRICING["local"]

    # Return pricing for known models
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Default to gpt-4o-mini pricing if model unknown (conservative estimate)
    print(f"Warning: Unknown model '{model}', using gpt-4o-mini pricing as estimate")
    return MODEL_PRICING["gpt-4o-mini"]


def estimate_compilation_cost(
    dataset_size: int,
    optimizer: str,
    student_model: str,
    teacher_model: str,
    input_fields: List[str],
    output_fields: List[str],
    instructions: str
) -> Dict[str, Any]:
    """
    Estimate the cost of compiling a DSPy program.

    Args:
        dataset_size: Number of examples in the training dataset
        optimizer: Type of optimizer (e.g., 'BootstrapFewShot', 'MIPROv2')
        student_model: Model used for the student program
        teacher_model: Model used for teaching/optimization
        input_fields: List of input field names
        output_fields: List of output field names
        instructions: Task instructions

    Returns:
        Dictionary containing cost estimate details
    """
    # Get pricing for both models
    student_pricing = get_model_pricing(student_model)
    teacher_pricing = get_model_pricing(teacher_model)

    # Estimate average token counts
    # Input: field names + instructions + example data (conservative estimate)
    avg_input_tokens = 100 + len(instructions.split()) * 1.3 + len(input_fields) * 20
    # Output: conservative estimate for generated response
    avg_output_tokens = 300

    # Estimate number of API calls based on optimizer type
    if optimizer == "LabeledFewShot":
        # No optimization calls, just inference
        teacher_calls = 0
        student_calls = 0
    elif optimizer == "BootstrapFewShot":
        # Teacher generates examples, student uses them
        teacher_calls = min(dataset_size * 3, 100)  # Cap at 100 calls
        student_calls = dataset_size  # One call per example for evaluation
    elif optimizer == "BootstrapFewShotWithRandomSearch":
        # More expensive due to random search
        teacher_calls = min(dataset_size * 10, 300)
        student_calls = dataset_size * 2
    elif optimizer == "MIPROv2":
        # Dataset-level optimization, not per-example
        teacher_calls = min(dataset_size + 100, 200)  # Bootstrap + optimization
        student_calls = dataset_size * 3  # Multiple evaluation rounds
    elif optimizer == "COPRO":
        # Similar to MIPRO but focused on prompts
        teacher_calls = min(dataset_size + 50, 150)
        student_calls = dataset_size * 2
    elif optimizer == "BootstrapFinetune":
        # Similar to BootstrapFewShot + finetuning overhead
        teacher_calls = min(dataset_size * 3, 100)
        student_calls = dataset_size
        # Note: Finetuning costs are separate and not included here
    else:
        # Default conservative estimate
        teacher_calls = dataset_size * 5
        student_calls = dataset_size

    # Calculate costs (convert from per-1M to actual cost)
    teacher_input_cost = (teacher_calls * avg_input_tokens * teacher_pricing["input"]) / 1_000_000
    teacher_output_cost = (teacher_calls * avg_output_tokens * teacher_pricing["output"]) / 1_000_000
    student_input_cost = (student_calls * avg_input_tokens * student_pricing["input"]) / 1_000_000
    student_output_cost = (student_calls * avg_output_tokens * student_pricing["output"]) / 1_000_000

    total_cost = teacher_input_cost + teacher_output_cost + student_input_cost + student_output_cost

    # Add 20% buffer for safety
    total_cost_with_buffer = total_cost * 1.2

    return {
        "estimated_cost_usd": round(total_cost_with_buffer, 4),
        "teacher_calls": teacher_calls,
        "student_calls": student_calls,
        "estimated_input_tokens": int((teacher_calls + student_calls) * avg_input_tokens),
        "estimated_output_tokens": int((teacher_calls + student_calls) * avg_output_tokens),
        "student_model": student_model,
        "teacher_model": teacher_model,
        "optimizer": optimizer,
        "dataset_size": dataset_size,
        "is_expensive": total_cost_with_buffer > 1.0,
        "is_very_expensive": total_cost_with_buffer > 5.0,
        "breakdown": {
            "teacher_input": round(teacher_input_cost, 4),
            "teacher_output": round(teacher_output_cost, 4),
            "student_input": round(student_input_cost, 4),
            "student_output": round(student_output_cost, 4)
        }
    }


def calculate_actual_cost(
    student_model: str,
    teacher_model: str
) -> Dict[str, Any]:
    """
    Calculate actual cost from DSPy LM history.

    Analyzes the DSPy call history to determine actual token usage and costs.

    Args:
        student_model: Model name used for the student program
        teacher_model: Model name used for teaching/optimization

    Returns:
        Dictionary containing:
        - actual_cost_usd: Total actual cost in USD
        - input_tokens: Total input tokens used
        - output_tokens: Total output tokens used
        - total_tokens: Sum of input and output tokens
        - breakdown: Dict with per-model cost breakdown
    """
    # Get model pricing
    student_pricing = get_model_pricing(student_model)
    teacher_pricing = get_model_pricing(teacher_model)

    # Initialize counters
    total_input_tokens = 0
    total_output_tokens = 0
    student_input_tokens = 0
    student_output_tokens = 0
    teacher_input_tokens = 0
    teacher_output_tokens = 0

    # Access DSPy history
    if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
        print("Warning: No LM configured in DSPy settings")
        return {
            "actual_cost_usd": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "breakdown": {}
        }

    history = dspy.settings.lm.history

    # Parse history to count tokens
    # DSPy history format: list of dicts with 'prompt', 'response', 'kwargs', etc.
    for call in history:
        # Try to extract token counts from the call
        # Different LM providers may store this differently
        input_tokens = 0
        output_tokens = 0

        # Try to get from response object (OpenAI format)
        if 'response' in call and hasattr(call['response'], 'usage'):
            usage = call['response'].usage
            if hasattr(usage, 'prompt_tokens'):
                input_tokens = usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                output_tokens = usage.completion_tokens

        # Fallback: estimate from text length if no usage data
        if input_tokens == 0 and 'prompt' in call:
            # Rough estimate: 1 token ≈ 4 characters
            input_tokens = len(str(call.get('prompt', ''))) // 4

        if output_tokens == 0 and 'response' in call:
            response_text = str(call.get('response', ''))
            if hasattr(call['response'], 'choices'):
                # OpenAI-style response
                try:
                    response_text = call['response'].choices[0].message.content
                except (AttributeError, IndexError, KeyError):
                    pass
            output_tokens = len(response_text) // 4

        # Add to totals
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens

        # For now, attribute all calls to student model
        # In future, could detect teacher calls by inspecting kwargs or model name
        student_input_tokens += input_tokens
        student_output_tokens += output_tokens

    # Calculate costs (convert from per-1M to actual)
    student_input_cost = (student_input_tokens * student_pricing["input"]) / 1_000_000
    student_output_cost = (student_output_tokens * student_pricing["output"]) / 1_000_000
    teacher_input_cost = (teacher_input_tokens * teacher_pricing["input"]) / 1_000_000
    teacher_output_cost = (teacher_output_tokens * teacher_pricing["output"]) / 1_000_000

    total_cost = student_input_cost + student_output_cost + teacher_input_cost + teacher_output_cost

    return {
        "actual_cost_usd": round(total_cost, 4),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "breakdown": {
            "student_input": round(student_input_cost, 4),
            "student_output": round(student_output_cost, 4),
            "teacher_input": round(teacher_input_cost, 4),
            "teacher_output": round(teacher_output_cost, 4)
        }
    }


# function to take a program from the program folder and run it on a row from the dataset
def generate_program_response(
    human_readable_id: str,
    row_data: Dict[str, Any],
    evaluate: bool = False
) -> Union[str, Tuple[str, Optional[float], Optional[str]]]:
    """
    Generate a response from a compiled program for a given row of data.

    Args:
        human_readable_id: The ID of the compiled program
        row_data: Dictionary containing input and optionally output (gold) data
        evaluate: If True, evaluate the prediction against gold data (if available)

    Returns:
        If evaluate=False: output string with prediction
        If evaluate=True: tuple of (output_string, evaluation_score, metric_type)
    """
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
    metric_type = program_details.get('metric_type', 'Exact Match')
    judge_prompt_id = program_details.get('judge_prompt_id')

    print("input_fields:", input_fields)
    print("output_fields:", output_fields)
    print("instructions:", instructions)
    print("input_descs:", input_descs)
    print("output_descs:", output_descs)
    print("dspy_module:", dspy_module)

    # Get optional parameters
    hint = program_details.get('hint')
    max_iters = program_details.get('max_iters', 3)

    # Create the custom signature
    CustomSignature = create_custom_signature(input_fields, output_fields, instructions, input_descs, output_descs)
    print("CustomSignature:", CustomSignature)
    compiled_program = create_dspy_module(dspy_module, CustomSignature, hint, max_iters)
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
        error_msg = f"Error: {str(e)}"
        if evaluate:
            return error_msg, None, None
        return error_msg

    # Prepare the output
    output = "Input:\n"
    for field in input_fields:
        output += f"{field}: {program_input[field]}\n"

    print("result:", result)

    output += "\nOutput:\n"
    for field in output_fields:
        output += f"{field}: {getattr(result, field)}\n"

    print("output:", output)

    # Optionally evaluate the prediction
    if evaluate:
        # Check if we have gold data for all output fields
        has_gold_data = all(field in row_data for field in output_fields)

        if has_gold_data:
            try:
                score = evaluate_single_prediction(
                    row_data, result, metric_type, output_fields, judge_prompt_id
                )
                return output, score, metric_type
            except Exception as e:
                print(f"Error during evaluation: {e}")
                return output, None, metric_type
        else:
            # No gold data available
            return output, None, metric_type

    return output


def create_evaluation_metric(
    metric_type: str,
    output_fields: List[str],
    judge_prompt_id: Optional[str] = None
) -> Callable[[Dict[str, Any], Any, Optional[Any]], float]:
    """
    Create an evaluation metric function based on the specified type.

    Args:
        metric_type: Type of metric ("Exact Match", "Cosine Similarity", or "LLM-as-a-Judge")
        output_fields: List of output field names to evaluate
        judge_prompt_id: ID of judge prompt (required for LLM-as-a-Judge)

    Returns:
        A metric function that takes (gold, pred, trace=None) and returns a score
    """
    if metric_type == "Exact Match":
        def metric(gold, pred, trace=None):
            if not pred or (isinstance(pred, dspy.Prediction) and not pred.__dict__):
                return 0
            try:
                return int(all(gold[field] == getattr(pred, field) for field in output_fields))
            except AttributeError:
                return 0

    elif metric_type == "Cosine Similarity":
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
            try:
                gold_vector = np.concatenate([get_embedding(str(gold[field])) for field in output_fields])
                pred_vector = np.concatenate([get_embedding(str(getattr(pred, field))) for field in output_fields])
                return cosine_similarity(gold_vector, pred_vector)
            except Exception as e:
                print(f"Error in cosine similarity metric: {e}")
                return 0

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

        judge_input_fields = judge_prompt_details.get('input_fields', [])
        judge_input_descs = judge_prompt_details.get('input_descs', [])
        judge_output_fields = judge_prompt_details.get('output_fields', [])
        judge_output_descs = judge_prompt_details.get('output_descs', [])
        judge_module = judge_prompt_details.get('dspy_module', 'Predict')
        judge_instructions = judge_prompt_details.get('instructions', '')
        judge_human_readable_id = judge_prompt_details.get('human_readable_id')

        # Create the custom signature for the judge program
        JudgeSignature = create_custom_signature(judge_input_fields, judge_output_fields,
                                                 judge_instructions, judge_input_descs, judge_output_descs)

        # Create the judge program
        judge_program = create_dspy_module(judge_module, JudgeSignature)

        # Load the compiled judge program
        if judge_prompt_id == example2_id:
            judge_program_path = f"example_data/{judge_human_readable_id}-program.json"
        else:
            judge_program_path = f"programs/{judge_human_readable_id}.json"

        if not os.path.exists(judge_program_path):
            raise ValueError(f"Compiled judge program not found: {judge_program_path}")

        judge_program.load(judge_program_path)

        def metric(gold, pred, trace=None):
            try:
                # Prepare input for the judge program
                judge_input = {}
                for field in judge_input_fields:
                    if field in gold:
                        judge_input[field] = gold[field]
                    elif hasattr(pred, field):
                        judge_input[field] = getattr(pred, field)
                    else:
                        judge_input[field] = ""

                # Run the judge program
                judge_result = judge_program(**judge_input)

                # Extract the score from the judge output
                score_field = judge_output_fields[0]
                score = float(getattr(judge_result, score_field))
                return score
            except Exception as e:
                print(f"Error in LLM-as-a-Judge metric: {e}")
                return 0
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")

    return metric


def evaluate_single_prediction(
    gold_data: Dict[str, Any],
    prediction: Any,
    metric_type: str,
    output_fields: List[str],
    judge_prompt_id: Optional[str] = None
) -> float:
    """
    Evaluate a single prediction against gold data using the specified metric.

    Args:
        gold_data: Dictionary containing the expected outputs
        prediction: The prediction object (dspy.Prediction or dict)
        metric_type: Type of metric to use
        output_fields: List of output field names
        judge_prompt_id: ID of judge prompt (for LLM-as-a-Judge)

    Returns:
        Numeric evaluation score
    """
    metric = create_evaluation_metric(metric_type, output_fields, judge_prompt_id)
    score = metric(gold_data, prediction, None)  # Pass None for trace parameter
    return float(score)


def save_consolidated_program(
    human_readable_id: str,
    prompt_config: Dict[str, Any],
    compiled_program_path: str,
    dataset: pd.DataFrame
) -> str:
    """
    Save all program data to a single consolidated .dspyui file.

    Args:
        human_readable_id: Unique identifier for the program
        prompt_config: Dictionary containing all prompt configuration
        compiled_program_path: Path to the compiled DSPy program JSON
        dataset: DataFrame containing the training data

    Returns:
        Path to the saved .dspyui file
    """
    import datetime

    # Create consolidated_programs directory if it doesn't exist
    os.makedirs("consolidated_programs", exist_ok=True)

    # Load the compiled program
    with open(compiled_program_path, 'r') as f:
        compiled_program_data = json.load(f)

    # Create consolidated structure
    consolidated_data = {
        "version": "1.0",
        "human_readable_id": human_readable_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "dspy_version": "3.0.3",
            "dspyui_version": "1.0"
        },
        "prompt_config": prompt_config,
        "compiled_program": compiled_program_data,
        "dataset": {
            "columns": list(dataset.columns),
            "data": dataset.to_dict('records')
        }
    }

    # Save to file
    output_path = f"consolidated_programs/{human_readable_id}.dspyui"
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return output_path


def load_consolidated_program(
    filepath: str
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Load program data from a consolidated .dspyui file.

    Args:
        filepath: Path to the .dspyui file

    Returns:
        Tuple of (prompt_config, compiled_program_data, dataset)
    """
    with open(filepath, 'r') as f:
        consolidated_data = json.load(f)

    # Validate version
    if consolidated_data.get("version") != "1.0":
        raise ValueError(f"Unsupported .dspyui version: {consolidated_data.get('version')}")

    # Extract components
    prompt_config = consolidated_data["prompt_config"]
    compiled_program_data = consolidated_data["compiled_program"]
    dataset = pd.DataFrame(
        data=consolidated_data["dataset"]["data"],
        columns=consolidated_data["dataset"]["columns"]
    )

    return prompt_config, compiled_program_data, dataset


def export_to_consolidated(human_readable_id: str) -> Optional[str]:
    """
    Export an existing program (saved in old 3-file format) to consolidated .dspyui format.

    Args:
        human_readable_id: ID of the program to export

    Returns:
        Path to the created .dspyui file, or None if any files are missing
    """
    # Check if all required files exist
    prompt_path = f"prompts/{human_readable_id}.json"
    program_path = f"programs/{human_readable_id}.json"
    dataset_path = f"datasets/{human_readable_id}.csv"

    if not all(os.path.exists(p) for p in [prompt_path, program_path, dataset_path]):
        return None

    # Load the data
    with open(prompt_path, 'r') as f:
        prompt_config = json.load(f)

    dataset = pd.read_csv(dataset_path)

    # Save as consolidated file
    output_path = save_consolidated_program(
        human_readable_id,
        prompt_config,
        program_path,
        dataset
    )

    return output_path


# ========================================
# Chain Building Functions
# ========================================

class ChainedProgram(dspy.Module):
    """
    A DSPy module that chains multiple modules together sequentially.
    Output from each step becomes input to the next step based on field mapping.
    """

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize a chained program.

        Args:
            steps: List of step configurations, each containing:
                - module: The DSPy module instance
                - input_mapping: Dict mapping chain inputs/previous outputs to module inputs
                - output_fields: List of output field names from this step
        """
        super().__init__()
        self.steps = steps

        # Store modules as attributes for DSPy to track them
        for i, step in enumerate(steps):
            setattr(self, f"step_{i}", step["module"])

    def forward(self, **inputs):
        """Execute the chain sequentially."""
        # Start with initial inputs
        context = dict(inputs)

        # Execute each step
        for i, step in enumerate(self.steps):
            module = step["module"]
            input_mapping = step["input_mapping"]

            # Prepare inputs for this step based on mapping
            step_inputs = {}
            for module_input, source in input_mapping.items():
                if source in context:
                    step_inputs[module_input] = context[source]
                else:
                    raise ValueError(f"Step {i}: Cannot find '{source}' in context. Available: {list(context.keys())}")

            # Execute this step
            result = module(**step_inputs)

            # Add outputs to context
            for field in step["output_fields"]:
                if hasattr(result, field):
                    context[field] = getattr(result, field)

        # Return final prediction with all accumulated context
        return dspy.Prediction(**context)


def create_chain_step(
    module_type: str,
    signature_str: str,
    instructions: str = ""
) -> dspy.Module:
    """
    Create a single step (module) for a chain.

    Args:
        module_type: Type of module ('Predict', 'ChainOfThought', etc.)
        signature_str: Signature string (e.g., 'input -> output')
        instructions: Optional instructions for the module

    Returns:
        Configured DSPy module
    """
    # Create signature with instructions if provided
    if instructions:
        signature = dspy.Signature(signature_str, instructions)
    else:
        signature = signature_str

    # Create module based on type
    if module_type == "Predict":
        return dspy.Predict(signature)
    elif module_type == "ChainOfThought":
        return dspy.ChainOfThought(signature)
    elif module_type == "ChainOfThoughtWithHint":
        return dspy.ChainOfThoughtWithHint(signature)
    elif module_type == "ProgramOfThought":
        return dspy.ProgramOfThought(signature)
    else:
        raise ValueError(f"Unsupported module type: {module_type}")


def compile_chain(
    chain_config: Dict[str, Any],
    dataset: pd.DataFrame,
    model: str,
    optimizer_name: str = "BootstrapFewShot",
    metric_type: str = "Exact Match",
    judge_prompt_id: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 8
) -> Tuple[ChainedProgram, Dict[str, Any], Dict[str, Any]]:
    """
    Compile a chained program using DSPy optimizers.

    Args:
        chain_config: Configuration dict containing:
            - name: Name of the chain
            - description: Description
            - initial_inputs: List of input field names for the entire chain
            - final_outputs: List of output field names from the final step
            - steps: List of step configs with module_type, signature, instructions, input_mapping
        dataset: Training data
        model: LLM model name
        optimizer_name: Name of DSPy optimizer
        metric_type: Evaluation metric type
        judge_prompt_id: Optional judge prompt ID for LLM-as-a-Judge
        provider: LLM provider
        api_key: API key
        llm_base_url: Base URL for local LLMs
        max_bootstrapped_demos: Max examples for bootstrap optimizers
        max_labeled_demos: Max examples for labeled optimizers

    Returns:
        Tuple of (compiled_chain, evaluation_results, cost_data)
    """
    # Configure LM
    configure_lm(model, provider, api_key, llm_base_url)

    # Create chain steps
    chain_steps = []
    for step_config in chain_config["steps"]:
        module = create_chain_step(
            step_config["module_type"],
            step_config["signature"],
            step_config.get("instructions", "")
        )

        # Parse output fields from signature
        sig_parts = step_config["signature"].split("->")
        if len(sig_parts) != 2:
            raise ValueError(f"Invalid signature: {step_config['signature']}")
        output_part = sig_parts[1].strip()
        output_fields = [f.strip() for f in output_part.split(",")]

        chain_steps.append({
            "module": module,
            "input_mapping": step_config["input_mapping"],
            "output_fields": output_fields
        })

    # Create chained program
    chain_program = ChainedProgram(chain_steps)

    # Prepare training data
    trainset = []
    for _, row in dataset.iterrows():
        example_dict = row.to_dict()
        trainset.append(dspy.Example(**example_dict).with_inputs(*chain_config["initial_inputs"]))

    # Split into train/dev sets
    split_point = int(len(trainset) * 0.8)
    train_examples = trainset[:split_point]
    dev_examples = trainset[split_point:] if split_point < len(trainset) else trainset[:2]

    # Create evaluation metric
    metric = create_evaluation_metric(metric_type, chain_config["final_outputs"], judge_prompt_id)

    # Configure and run optimizer
    if optimizer_name == "LabeledFewShot":
        optimizer = LabeledFewShot(k=min(max_labeled_demos, len(train_examples)))
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
    elif optimizer_name == "BootstrapFewShot":
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=min(max_bootstrapped_demos, len(train_examples))
        )
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
    elif optimizer_name == "BootstrapFewShotWithRandomSearch":
        optimizer = BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=min(max_bootstrapped_demos, len(train_examples)),
            num_candidate_programs=3
        )
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
    elif optimizer_name == "MIPROv2":
        optimizer = MIPROv2(
            metric=metric,
            num_candidates=3,
            init_temperature=0.7
        )
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples, num_trials=5, max_bootstrapped_demos=max_bootstrapped_demos, max_labeled_demos=max_labeled_demos)
    elif optimizer_name == "COPRO":
        optimizer = COPRO(metric=metric, breadth=3, depth=1)
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples, eval_kwargs={"num_threads": 1})
    elif optimizer_name == "BootstrapFinetune":
        optimizer = BootstrapFinetune(metric=metric)
        compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Evaluate on dev set
    evaluator = Evaluate(devset=dev_examples, metric=metric, num_threads=1, display_progress=True)
    score = evaluator(compiled_chain)

    # Calculate cost
    cost_data = calculate_actual_cost(dspy.settings.lm.history, model)

    evaluation_results = {
        "score": score,
        "dev_set_size": len(dev_examples),
        "train_set_size": len(train_examples)
    }

    return compiled_chain, evaluation_results, cost_data


def save_chain_program(
    human_readable_id: str,
    chain_config: Dict[str, Any],
    compiled_chain: ChainedProgram,
    dataset: pd.DataFrame,
    evaluation_results: Dict[str, Any],
    cost_data: Dict[str, Any],
    model: str,
    optimizer_name: str,
    metric_type: str
) -> str:
    """
    Save a compiled chain program and its configuration.

    Args:
        human_readable_id: Unique ID for the chain
        chain_config: Chain configuration dict
        compiled_chain: Compiled ChainedProgram
        dataset: Training dataset
        evaluation_results: Evaluation results dict
        cost_data: Cost tracking data
        model: Model name
        optimizer_name: Optimizer name
        metric_type: Metric type

    Returns:
        Path to saved consolidated file
    """
    # Create directories
    os.makedirs("chains", exist_ok=True)
    os.makedirs("consolidated_programs", exist_ok=True)

    # Save chain program
    chain_path = f"chains/{human_readable_id}.json"
    compiled_chain.save(chain_path)

    # Create metadata
    metadata = {
        "human_readable_id": human_readable_id,
        "chain_name": chain_config["name"],
        "description": chain_config.get("description", ""),
        "model": model,
        "optimizer": optimizer_name,
        "metric": metric_type,
        "evaluation_score": evaluation_results["score"],
        "train_set_size": evaluation_results["train_set_size"],
        "dev_set_size": evaluation_results["dev_set_size"],
        "actual_cost": cost_data.get("actual_cost_usd", 0),
        "total_tokens": cost_data.get("total_tokens", 0),
        "input_tokens": cost_data.get("input_tokens", 0),
        "output_tokens": cost_data.get("output_tokens", 0),
        "created_at": datetime.datetime.now().isoformat(),
        "chain_config": chain_config
    }

    # Save metadata
    metadata_path = f"chains/{human_readable_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save consolidated file
    with open(chain_path, 'r') as f:
        compiled_chain_data = json.load(f)

    consolidated_data = {
        "version": "1.0",
        "type": "chain",
        "human_readable_id": human_readable_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "dspy_version": "3.0.3",
            "dspyui_version": "1.0"
        },
        "chain_config": chain_config,
        "model_config": {
            "model": model,
            "optimizer": optimizer_name,
            "metric": metric_type
        },
        "evaluation_results": evaluation_results,
        "cost_data": cost_data,
        "compiled_program": compiled_chain_data,
        "dataset": {
            "columns": list(dataset.columns),
            "data": dataset.to_dict('records')
        }
    }

    consolidated_path = f"consolidated_programs/{human_readable_id}.dspyui"
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return consolidated_path


def execute_chain(
    human_readable_id: str,
    inputs: Dict[str, Any]
) -> Tuple[dspy.Prediction, Optional[float]]:
    """
    Execute a saved chain program on new inputs.

    Args:
        human_readable_id: ID of the saved chain
        inputs: Dict of input values

    Returns:
        Tuple of (prediction, evaluation_score)
    """
    # Load chain
    chain_path = f"chains/{human_readable_id}.json"
    metadata_path = f"chains/{human_readable_id}_metadata.json"

    if not os.path.exists(chain_path):
        raise FileNotFoundError(f"Chain not found: {chain_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create chain structure from config
    chain_config = metadata["chain_config"]
    chain_steps = []
    for step_config in chain_config["steps"]:
        module = create_chain_step(
            step_config["module_type"],
            step_config["signature"],
            step_config.get("instructions", "")
        )

        sig_parts = step_config["signature"].split("->")
        output_part = sig_parts[1].strip()
        output_fields = [f.strip() for f in output_part.split(",")]

        chain_steps.append({
            "module": module,
            "input_mapping": step_config["input_mapping"],
            "output_fields": output_fields
        })

    # Load compiled chain
    chain_program = ChainedProgram(chain_steps)
    chain_program.load(chain_path)

    # Execute
    prediction = chain_program(**inputs)

    return prediction, None


def list_chains(
    name_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all saved chain programs.

    Args:
        name_filter: Optional filter string for chain names

    Returns:
        List of chain metadata dicts
    """
    if not os.path.exists('chains'):
        return []

    chains = []
    for file in os.listdir('chains'):
        if file.endswith('_metadata.json'):
            with open(os.path.join('chains', file), 'r') as f:
                metadata = json.load(f)

                if name_filter and name_filter.lower() not in metadata["chain_name"].lower():
                    continue

                chains.append({
                    "ID": metadata["human_readable_id"],
                    "Name": metadata["chain_name"],
                    "Description": metadata.get("description", ""),
                    "Steps": len(metadata["chain_config"]["steps"]),
                    "Eval Score": metadata.get("evaluation_score", "N/A"),
                    "Cost": metadata.get("actual_cost", 0),
                    "Tokens": metadata.get("total_tokens", 0),
                    "Model": metadata.get("model", ""),
                    "Details": json.dumps(metadata, indent=2)
                })

    return chains
