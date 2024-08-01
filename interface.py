import gradio as gr
import json
import pandas as pd

from helpers import compile_program, load_csv

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# DSPy Program Compiler")
    gr.Markdown("Compile a DSPy program by specifying parameters and example data.")
    
    input_count = gr.State(1)
    output_count = gr.State(1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Inputs")
            add_input_btn = gr.Button("Add Input Field")
            add_input_btn.click(lambda count: count + 1, input_count, input_count)
        with gr.Column():
            gr.Markdown("### Outputs")
            add_output_btn = gr.Button("Add Output Field")
            add_output_btn.click(lambda count: count + 1, output_count, output_count)

    @gr.render(inputs=[input_count, output_count])
    def render_tracks(input_count, output_count):
        inputs = []
        outputs = []
        with gr.Row():
            with gr.Column():
                for i in range(input_count):
                    with gr.Group():
                        input = gr.Textbox(placeholder=f"Input{i+1}", key=f"input-{i}", show_label=False)
                        inputs.append(input)
            
            with gr.Column():
                for i in range(output_count):
                    with gr.Group():
                        output = gr.Textbox(placeholder=f"Output{i+1}", key=f"output-{i}", show_label=False)
                        outputs.append(output)

        gr.Markdown("### Settings")
        with gr.Row():
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-opus", "claude-3-haiku", "claude-3.5-sonnet"]
            llm_model = gr.Dropdown(model_options, label="Model", value="gpt-4o-mini")
            teacher_model = gr.Dropdown(model_options, label="Teacher", value="gpt-4o")
            dspy_module = gr.Dropdown(["Predict", "ChainOfThought", "MultiChainComparison"], label="Module", value="Predict")
        with gr.Row():
            optimizer = gr.Dropdown(["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "COPRO", "MIPRO"], label="Optimizer", value="BootstrapFewShotWithRandomSearch")
            metric_type = gr.Radio(["Exact Match", "LLM-as-a-Judge"], label="Metric", value="Exact Match")

        gr.Markdown("### Data")
        with gr.Column():
            # TODO: make this the actual values inputted instead of the Input numbers
            example_data = gr.Dataframe(
                headers=[f"Input{i+1}" for i in range(input_count)] + [f"Output{i+1}" for i in range(output_count)],
                datatype=["str"] * (input_count + output_count),
                label="Example Data"
            )
            csv_file = gr.File(label="Or Upload CSV")

        def compile(data):
            print(data)
            input_fields = [data[input] for input in inputs if data[input].strip()]
            output_fields = [data[output] for output in outputs if data[output].strip()]
            
            if csv_file is not None:
                filename = csv_file.name or ""
                example_data = load_csv(filename)
            else:
                example_data = example_data.to_dict('records')
            
            try:
                result = compile_program(input_fields, output_fields, llm_model, teacher_model, dspy_module, example_data, metric_type, optimizer)
            except Exception as e:
                result = f"An error occurred: {str(e)}"
            
            signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            
            return result, signature

        compile_button.click(
            compile,
            inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, csv_file, metric_type, optimizer]),
            outputs=[output, signature]
        )
    
    compile_button = gr.Button("Compile Program")
    output = gr.Textbox(label="Compiled Program")
    signature = gr.Textbox(label="Signature", interactive=False)

# Launch the interface
iface.launch()