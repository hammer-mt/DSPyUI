import gradio as gr
import dspy
import pandas as pd
import json
from typing import List, Dict, Any
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPRO, BootstrapFinetune, COPRO
import os

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

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# DSPy Program Compiler")
    gr.Markdown("Compile a DSPy program by specifying parameters and example data.")
    
    input_count = gr.State(1)
    output_count = gr.State(1)
    
    with gr.Row():
        with gr.Column():
            input_container = gr.Column()
            with gr.Row():
                add_input = gr.Button("Add Input Field")
                remove_input = gr.Button("Remove Input Field")
        
        with gr.Column():
            output_container = gr.Column()
            with gr.Row():
                add_output = gr.Button("Add Output Field")
                remove_output = gr.Button("Remove Output Field")
    
    llm_model = gr.Dropdown(["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-opus", "claude-3-haiku", "claude-3.5-sonnet"], label="LLM Model")
    dspy_module = gr.Dropdown(["Predict", "ChainOfThought", "ReAct"], label="DSPy Module")
    example_data = gr.Textbox(label="Example Data (JSON format)")
    csv_file = gr.File(label="Or Upload CSV")
    metric_type = gr.Radio(["Exact Match"], label="Metric Type")
    optimizer = gr.Dropdown(["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPRO", "BootstrapFinetune", "COPRO"], label="Optimizer")
    
    compile_button = gr.Button("Compile Program")
    output = gr.Textbox(label="Compiled Program")
    
    def render_fields(count, prefix):
        return [gr.Textbox(label=f"{prefix} Field {i+1}", value=f"{prefix.lower()}{i+1}", key=f"{prefix.lower()}-{i}") for i in range(count)]

    def update_input_fields(count):
        return count, gr.Column.update(children=render_fields(count, "Input"))

    def update_output_fields(count):
        return count, gr.Column.update(children=render_fields(count, "Output"))

    def add_input_field(count):
        return update_input_fields(count + 1)

    def remove_input_field(count):
        return update_input_fields(max(1, count - 1))

    def add_output_field(count):
        return update_output_fields(count + 1)

    def remove_output_field(count):
        return update_output_fields(max(1, count - 1))

    add_input.click(add_input_field, inputs=[input_count], outputs=[input_count, input_container])
    remove_input.click(remove_input_field, inputs=[input_count], outputs=[input_count, input_container])
    add_output.click(add_output_field, inputs=[output_count], outputs=[output_count, output_container])
    remove_output.click(remove_output_field, inputs=[output_count], outputs=[output_count, output_container])
    
    def gradio_interface(input_fields, output_fields, llm_model, dspy_module, example_data, csv_file, metric_type, optimizer):
        input_fields = [field.strip() for field in input_fields if field.strip()]
        output_fields = [field.strip() for field in output_fields if field.strip()]
        
        if csv_file is not None:
            example_data = load_csv(csv_file.name)
        else:
            example_data = json.loads(example_data)
        
        try:
            result = compile_program(input_fields, output_fields, llm_model, dspy_module, example_data, metric_type, optimizer)
        except Exception as e:
            result = f"An error occurred: {str(e)}"
        
        return result

    compile_button.click(
        gradio_interface,
        inputs=[
            input_container,
            output_container,
            llm_model, dspy_module, example_data, csv_file, metric_type, optimizer
        ],
        outputs=[output]
    )

    # Initial render of input and output fields
    input_container.render(lambda: render_fields(1, "Input"))
    output_container.render(lambda: render_fields(1, "Output"))

# Launch the interface
iface.launch()