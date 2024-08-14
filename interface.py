import gradio as gr
import pandas as pd
import json
import os

from core import compile_program

# Function to list prompts
def list_prompts():
    if not os.path.exists('prompts'):
        return []
    files = os.listdir('prompts')
    if not files:
        return []
    
    prompt_details = []
    for file in files:
        print("file: ", file)
        if file.endswith('.json'):
            with open(os.path.join('prompts', file), 'r') as f:
                data = json.load(f)
                prompt_id = file
                signature = data.get('signature', 'N/A')
                eval_score = data.get('evaluation_score', 'N/A')
                # Exclude example data
                details = {k: v for k, v in data.items() if k != 'example_data'}
                prompt_details.append({
                    "ID": prompt_id,
                    "Signature": signature,
                    "Eval Score": eval_score,
                    "Details": json.dumps(details, indent=4)  # Add full details as a JSON string
                })
    
    return prompt_details  # Return the list of prompts as dictionaries

# Gradio interface
with gr.Blocks() as iface:
    with gr.Tabs():
        with gr.TabItem("Compile Program"):
            gr.Markdown("# DSPyUI: a Gradio user interface for DSPy")
            gr.Markdown("Compile a DSPy program by specifying your settings and providing example data.")
            
            # Task Instructions
            with gr.Row():
                with gr.Column(scale=4):
                    instructions = gr.Textbox(
                        label="Task Instructions",
                        lines=3,
                        placeholder="Enter detailed task instructions here.",
                        info="Provide clear and comprehensive instructions for the task. This will guide the DSPy program in understanding the specific requirements and expected outcomes."
                    )

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
                                input = gr.Textbox(
                                    placeholder=input_value,
                                    key=f"input-{i}",
                                    show_label=False,
                                    label=f"Input {i+1}",
                                    info="Specify the name and description of this input field."
                                )
                                inputs.append(input)
                    
                    with gr.Column():
                        for i, output_value in enumerate(output_values):
                            with gr.Group():
                                output = gr.Textbox(
                                    placeholder=output_value,
                                    key=f"output-{i}",
                                    show_label=False,
                                    label=f"Output {i+1}",
                                    info="Specify the name and description of this output field."
                                )
                                outputs.append(output)

                gr.Markdown("### Settings")
                with gr.Row():
                    model_options = [
                        "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini",
                        "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
                        "claude-3-sonnet-20240229", "claude-3-haiku-20240307",
                        "mixtral-8x7b-32768", "gemma-7b-it", "llama3-70b-8192",
                        "llama3-8b-8192", "gemma2-9b-it"
                    ]
                    llm_model = gr.Dropdown(
                        model_options,
                        label="Model",
                        value="gpt-4o-mini",
                        info="Select the main language model for your DSPy program. This model will be used for inference. Typically you want to choose a fast and cheap model here, and train it on your task to improve quality."
                    )
                    teacher_model = gr.Dropdown(
                        model_options,
                        label="Teacher",
                        value="gpt-4o",
                        info="Select a more capable (but slower and more expensive) model to act as a teacher during the compilation process. This model helps generate high-quality examples and refine prompts."
                    )
                    dspy_module = gr.Dropdown(
                        ["Predict", "ChainOfThought"],
                        label="Module",
                        value="Predict",
                        info="Choose the DSPy module that best fits your task. Predict is for simple tasks, ChainOfThought for complex reasoning."
                    )
                with gr.Row():
                    optimizer = gr.Dropdown(
                        ["None", "BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPRO", "MIPROv2", "COPRO"],
                        label="Optimizer",
                        value="None",
                        info="Choose optimization strategy: None (no optimization), BootstrapFewShot (small datasets, ~10 examples) uses few-shot learning; BootstrapFewShotWithRandomSearch (medium, ~50) adds randomized search; MIPRO, MIPROv2, and COPRO (large, 300+) also optimize the prompt instructions."
                    )
                    metric_type = gr.Radio(
                        ["Exact Match", "LLM-as-a-Judge"],
                        label="Metric",
                        value="Exact Match",
                        info="Choose how to evaluate your program's performance. Exact Match is suitable for tasks with clear correct answers, while LLM-as-a-Judge is better for open-ended or subjective tasks. Note: you must have another compiled program (typically a program trained on a classification task) to use LLM-as-a-Judge."
                    )

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
                        visible=False,
                        label="Example Data"
                    )
                    
                    export_csv_btn = gr.Button("Export to CSV", visible=False)
                    csv_download = gr.File(label="Download CSV", visible=False)
                    error_message = gr.Markdown()

                compile_button = gr.Button("Compile Program", visible=False, variant="primary")
                with gr.Column() as compilation_results:
                    gr.Markdown("### Results")
                    signature = gr.Textbox(label="Signature", interactive=False, info="The compiled signature of your DSPy program, showing inputs and outputs.")
                    evaluation_score = gr.Number(label="Evaluation Score", info="The evaluation score of your compiled DSPy program.")
                    optimized_prompt = gr.Textbox(label="Optimized Prompt", info="The optimized prompt generated by the DSPy compiler for your program.")
                    usage_instructions = gr.Textbox(label="Usage Instructions", info="Instructions on how to use your compiled DSPy program.")

                def show_dataframe(*args):
                    data = {f"input-{i}": value for i, value in enumerate(args[:len(input_values)])}
                    data.update({f"output-{i}": value for i, value in enumerate(args[len(input_values):])})
                    
                    input_fields = [value for key, value in data.items() if key.startswith("input-") and value and value.strip()]
                    output_fields = [value for key, value in data.items() if key.startswith("output-") and value and value.strip()]

                    headers = input_fields + output_fields
                    
                    # Create a new dataframe with the correct headers
                    new_df = pd.DataFrame(columns=headers)
                    
                    return gr.update(visible=True, value=new_df), gr.update(visible=True), gr.update(visible=True)

                enter_manually_btn.click(
                    show_dataframe,
                    inputs=inputs + outputs,
                    outputs=[example_data, export_csv_btn, compile_button]
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
                                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: CSV headers do not match expected format. Expected: {expected_headers}, Got: {list(df.columns)}")
                            return df, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                        except Exception as e:
                            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: {str(e)}")
                    return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

                upload_csv_btn.upload(
                    process_csv,
                    inputs=[upload_csv_btn] + inputs + outputs,
                    outputs=[example_data, example_data, compile_button, error_message]
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
                    input_fields = [data[input] for input in inputs if data[input].strip()]
                    output_fields = [data[output] for output in outputs if data[output].strip()]

                    usage_instructions, optimized_prompt = compile_program(
                        input_fields,
                        output_fields,
                        data[dspy_module],
                        data[llm_model], 
                        data[teacher_model],
                        data[example_data],
                        data[optimizer],
                        data[instructions]
                    )
                    
                    signature = f"{', '.join(input_fields)} -> {', '.join(output_fields)}"
                    
                    # Extract evaluation score from usage_instructions
                    score_line = [line for line in usage_instructions.split('\n') if line.startswith("Evaluation score:")][0]
                    evaluation_score = float(score_line.split(":")[1].strip())
                    
                    # Remove the evaluation score line from usage_instructions
                    usage_instructions = '\n'.join([line for line in usage_instructions.split('\n') if not line.startswith("Evaluation score:")])

                    # Extract human-readable ID from usage_instructions
                    human_readable_id = None
                    for line in usage_instructions.split('\n'):
                        if "programs/" in line and ".json" in line:
                            human_readable_id = line.split('programs/')[1].split('.json')[0]
                            break
                    
                    if human_readable_id is None:
                        raise ValueError("Could not extract human-readable ID from usage instructions")

                    # Save details to JSON
                    details = {
                        "input_fields": input_fields,
                        "output_fields": output_fields,
                        "dspy_module": data[dspy_module],
                        "llm_model": data[llm_model],
                        "teacher_model": data[teacher_model],
                        "optimizer": data[optimizer],
                        "instructions": data[instructions],
                        "signature": signature,
                        "evaluation_score": evaluation_score,
                        "optimized_prompt": optimized_prompt,
                        "usage_instructions": usage_instructions,
                        "human_readable_id": human_readable_id
                    }
                    
                    # Create 'prompts' folder if it doesn't exist
                    if not os.path.exists('prompts'):
                        os.makedirs('prompts')
                    
                    # Save JSON file with human-readable ID
                    json_filename = f"prompts/{human_readable_id}.json"
                    with open(json_filename, 'w') as f:
                        json.dump(details, f, indent=4)
                    return signature, evaluation_score, optimized_prompt, usage_instructions

                compile_button.click(
                    compile,
                    inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, upload_csv_btn, optimizer, instructions]),
                    outputs=[signature, evaluation_score, optimized_prompt, usage_instructions]
                )

        with gr.TabItem("View Prompts"):
            gr.Markdown("# View Prompts")
            prompts = list_prompts()
            
            selected_prompt_details = gr.Markdown()  # Placeholder for selected prompt details
            
            def show_prompt_details(prompt):
                details = json.loads(prompt["Details"])
                formatted_details = "\n".join([f"**{key}:** {value}" for key, value in details.items()])
                return gr.update(value=formatted_details)
            
            # Create rows with 4 columns each
            num_columns = 4
            for i in range(0, len(prompts), num_columns):
                with gr.Row():
                    for j in range(num_columns):
                        if i + j < len(prompts):
                            prompt = prompts[i + j]
                            with gr.Column(scale=1):
                                with gr.Group():
                                    gr.Markdown(f"**ID:** {prompt['ID']}")
                                    gr.Markdown(f"**Signature:** {prompt['Signature']}")
                                    gr.Markdown(f"**Eval Score:** {prompt['Eval Score']}")
                                    gr.Button("View Details").click(
                                        show_prompt_details,
                                        inputs=[gr.State(prompt)],
                                        outputs=[selected_prompt_details]
                                    )
                                    gr.Markdown("---")  # Add a horizontal line for separation

            gr.Markdown("## Prompt Details")
            selected_prompt_details  # Display the selected prompt details here

# Launch the interface
iface.launch()