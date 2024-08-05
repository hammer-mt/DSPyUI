import gradio as gr
import json
import pandas as pd
import io
from dspy.signatures import InputField, OutputField

from core import compile_program, load_csv

# Gradio interface
with gr.Blocks() as iface:
    gr.Markdown("# DSPy Program Compiler")
    gr.Markdown("Compile a DSPy program by specifying parameters and example data.")
    
    # Add the instructions textbox here, at the top of the interface
    instructions = gr.Textbox(label="Task Instructions", lines=3, placeholder="Enter task instructions here...")
    
    input_values = gr.State(["Input1"])
    output_values = gr.State(["Output1"])

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Inputs")
            with gr.Row():
                add_input_btn = gr.Button("Add Input Field")
                del_input_btn = gr.Button("Delete Last Input")
            add_input_btn.click(
                lambda values: values + [f"Input{len(values)+1}"],
                inputs=input_values,
                outputs=input_values
            )
            del_input_btn.click(
                lambda values: values[:-1] if len(values) > 1 else values,
                inputs=input_values,
                outputs=input_values
            )
        with gr.Column():
            gr.Markdown("### Outputs")
            with gr.Row():
                add_output_btn = gr.Button("Add Output Field")
                del_output_btn = gr.Button("Delete Last Output")
            add_output_btn.click(
                lambda values: values + [f"Output{len(values)+1}"],
                inputs=output_values,
                outputs=output_values
            )
            del_output_btn.click(
                lambda values: values[:-1] if len(values) > 1 else values,
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
                        input = gr.Textbox(placeholder=input_value, key=f"input-{i}", show_label=False, label=f"Input {i+1}")
                        inputs.append(input)
            
            with gr.Column():
                for i, output_value in enumerate(output_values):
                    with gr.Group():
                        output = gr.Textbox(placeholder=output_value, key=f"output-{i}", show_label=False, label=f"Output {i+1}")
                        outputs.append(output)

        gr.Markdown("### Settings")
        with gr.Row():
            model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "claude-3-sonnet", "claude-3-opus", "claude-3-haiku", "claude-3.5-sonnet"]
            llm_model = gr.Dropdown(model_options, label="Model", value="gpt-4o-mini")
            teacher_model = gr.Dropdown(model_options, label="Teacher", value="gpt-4o")
            dspy_module = gr.Dropdown(["Predict", "ChainOfThought", "MultiChainComparison"], label="Module", value="Predict")
        with gr.Row():
            optimizer = gr.Dropdown(["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "COPRO", "MIPRO"], label="Optimizer", value="BootstrapFewShot")
            metric_type = gr.Radio(["Exact Match", "LLM-as-a-Judge"], label="Metric", value="Exact Match")

        gr.Markdown("### Data")
        with gr.Column():
            with gr.Row():
                enter_manually_btn = gr.Button("Enter manually")
                upload_csv_btn = gr.UploadButton("Upload CSV", file_types=[".csv"])
            
            example_data = gr.Dataframe(
                headers=input_values + output_values,
                datatype=["str"] * (len(input_values) + len(output_values)),
                interactive=True,
                row_count=1,
                col_count=(len(input_values) + len(output_values), "fixed"),
                visible=False
            )
            export_csv_btn = gr.Button("Export to CSV", visible=False)
            csv_download = gr.File(label="Download CSV", visible=False)
            error_message = gr.Markdown()

        compile_button = gr.Button("Compile Program", visible=False, variant="primary")
        result = gr.Textbox(label="Optimization Result", visible=False)
        signature = gr.Textbox(label="Signature", interactive=False, visible=False)

        def show_dataframe(*args):
            data = {f"input-{i}": value for i, value in enumerate(args[:len(input_values)])}
            data.update({f"output-{i}": value for i, value in enumerate(args[len(input_values):])})
            
            print("EXAMPLE DATA:\n")
            print(data)
            print("---")
            input_fields = [value for key, value in data.items() if key.startswith("input-") and value and value.strip()]
            output_fields = [value for key, value in data.items() if key.startswith("output-") and value and value.strip()]
            print("INPUT FIELDS:\n")
            print(input_fields)
            print("---")
            print("OUTPUT FIELDS:\n")
            print(output_fields)
            print("---")
            headers = input_fields + output_fields
            print("HEADERS:\n")
            print(headers)
            print("---")
            
            # Create a new dataframe with the correct headers
            new_df = pd.DataFrame(columns=headers)
            
            return gr.update(visible=True, value=new_df), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

        enter_manually_btn.click(
            show_dataframe,
            inputs=inputs + outputs,
            outputs=[example_data, export_csv_btn, compile_button, result, signature]
        )

        def process_csv(file, *args):
            if file is not None:
                try:
                    df = pd.read_csv(file.name)
                    # Use the actual input and output names entered by the user
                    input_fields = [arg for arg in args[:len(input_values)] if arg and arg.strip()]
                    output_fields = [arg for arg in args[len(input_values):] if arg and arg.strip()]
                    expected_headers = input_fields + output_fields
                    
                    if list(df.columns) != expected_headers:
                        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: CSV headers do not match expected format. Expected: {expected_headers}, Got: {list(df.columns)}")
                    return df, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                except Exception as e:
                    return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: {str(e)}")
            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

        upload_csv_btn.upload(
            process_csv,
            inputs=[upload_csv_btn] + inputs + outputs,
            outputs=[example_data, example_data, export_csv_btn, compile_button, result, signature, error_message]
        )

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


            result = compile_program(
                input_fields,
                output_fields,
                data[dspy_module],
                data[llm_model], 
                data[teacher_model],
                data[example_data],
                data[optimizer],
                data[instructions]  # Add this line to pass instructions to compile_program
            )

            
            signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
            
            return result, signature

        compile_button.click(
            compile,
            inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, upload_csv_btn, optimizer, instructions]),  # Add instructions here
            outputs=[result, signature]
        )

# Launch the interface
iface.launch()