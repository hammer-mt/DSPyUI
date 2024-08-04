import gradio as gr
import json
import pandas as pd
import io

from core import compile_program, load_csv

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# DSPy Program Compiler")
    gr.Markdown("Compile a DSPy program by specifying parameters and example data.")
    
    input_values = gr.State(["Input1"])
    output_values = gr.State(["Output1"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Inputs")
            add_input_btn = gr.Button("Add Input Field")
            add_input_btn.click(
                lambda values: values + [f"Input{len(values)+1}"],
                inputs=input_values,
                outputs=input_values
            )
        with gr.Column():
            gr.Markdown("### Outputs")
            add_output_btn = gr.Button("Add Output Field")
            add_output_btn.click(
                lambda values: values + [f"Output{len(values)+1}"],
                inputs=output_values,
                outputs=output_values
            )

    @gr.render(inputs=[input_values, output_values])
    def render_variables(input_values, output_values):
        inputs = []
        outputs = []
        with gr.Row():
            with gr.Column():
                for i, input_value in enumerate(input_values):
                    with gr.Group():
                        input = gr.Textbox(placeholder=input_value, key=f"input-{i}", show_label=True, label=f"Input {i+1}")
                        inputs.append(input)
            
            with gr.Column():
                for i, output_value in enumerate(output_values):
                    with gr.Group():
                        output = gr.Textbox(placeholder=output_value, key=f"output-{i}", show_label=True, label=f"Output {i+1}")
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
                headers=input_values + output_values,
                datatype=["str"] * (len(input_values) + len(output_values)),
                label="Example Data",
                interactive=True,
                row_count=1,
                col_count=(len(input_values) + len(output_values), "fixed")
            )
            export_csv_btn = gr.Button("Export to CSV")
            csv_download = gr.File(label="Download CSV", visible=False)
            csv_file = gr.File(label="Or Upload CSV")

        def export_to_csv(data):
            df = pd.DataFrame(data)
            filename = "exported_data.csv"
            df.to_csv(filename, index=False)
            return filename

        

        export_csv_btn.click(
            export_to_csv,
            inputs=[example_data],
            outputs=[csv_download]
        ).then(
            lambda: gr.update(visible=True),
            outputs=[csv_download]
        )

        def compile(data):
            print("DATA:\n")
            print(data)
            print("---")

            print("EXAMPLE DATA:\n")
            print(data[example_data])
            print("---")
            input_fields = [data[input] for input in inputs if data[input].strip()]
            output_fields = [data[output] for output in outputs if data[output].strip()]
        
            # try:
            #     result = compile_program(input_fields, output_fields, llm_model, teacher_model, dspy_module, example_data_records, metric_type, optimizer)
            # except Exception as e:
            #     result = f"An error occurred: {str(e)}"
            
            signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            
            return signature

        compile_button.click(
            compile,
            inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, csv_file, metric_type, optimizer]),
            outputs=[signature]
        )
    
    compile_button = gr.Button("Compile Program")
    output = gr.Textbox(label="Compiled Program")
    signature = gr.Textbox(label="Signature", interactive=False)

# Launch the interface
iface.launch()