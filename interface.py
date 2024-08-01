import gradio as gr
import json

from helpers import compile_program, load_csv

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