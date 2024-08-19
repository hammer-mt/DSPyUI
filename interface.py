import gradio as gr
import pandas as pd
import json
import os
import glob

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

# Function to get available prompts for LLM-as-a-Judge
def get_available_prompts():
    prompt_files = glob.glob('prompts/*.json')
    prompts = []
    for file in prompt_files:
        with open(file, 'r') as f:
            data = json.load(f)
            prompts.append({
                "id": os.path.basename(file).split('.')[0],
                "signature": data.get('signature', 'N/A'),
                "eval_score": data.get('evaluation_score', 'N/A')
            })
    return prompts


# Gradio interface
custom_css = """
.expand-button {
  min-width: 20px !important;
  width: 20px !important;
  padding: 0 !important;
  font-size: 10px !important;
}
.prompt-card {
  height: 150px !important;
  display: flex !important;
  flex-direction: column !important;
  justify-content: space-between !important;
  padding: 10px !important;
  position: relative !important;
}
.prompt-details {
  flex-grow: 1 !important;
}
.view-details-btn {
  position: absolute !important;
  bottom: 10px !important;
  right: 10px !important;
}
"""

with gr.Blocks(css=custom_css) as iface:
    def create_prompt_element(prompt):
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(f"**ID:** {prompt['ID']}")
                gr.Markdown(f"**Signature:** {prompt['Signature']}")
                gr.Markdown(f"**Eval Score:** {prompt['Eval Score']}")
                view_details_btn = gr.Button("View Details")
                gr.Markdown("---")  # Add a horizontal line for separation
        return gr.update(visible=True)

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
                        remove_input_btn = gr.Button("Remove Last Input")
                with gr.Column():
                    gr.Markdown("### Outputs")
                    with gr.Row():  
                        add_output_btn = gr.Button("Add Output Field")
                        remove_output_btn = gr.Button("Remove Last Output")

            add_input_btn.click(
                lambda values: values + [f"Input{len(values)+1}"],
                inputs=input_values,
                outputs=input_values
            )

            remove_input_btn.click(
                lambda values: values[:-1] if values else values,
                inputs=input_values,
                outputs=input_values
            )

            add_output_btn.click(
                lambda values: values + [f"Output{len(values)+1}"],
                inputs=output_values,
                outputs=output_values
            )

            remove_output_btn.click(
                lambda values: values[:-1] if values else values,
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
                                with gr.Row():
                                    input_name = gr.Textbox(
                                        placeholder=input_value,
                                        key=f"input-name-{i}",
                                        show_label=False,
                                        label=f"Input {i+1} Name",
                                        info="Specify the name of this input field.",
                                        scale=9
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                input_desc = gr.Textbox(
                                    placeholder="Description (optional)",
                                    key=f"input-desc-{i}",
                                    show_label=False,
                                    label=f"Input {i+1} Description",
                                    info="Optionally provide a description for this input field.",
                                    visible=False
                                )
                                desc_visible = gr.State(False)
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, input_desc]
                                )
                                inputs.extend([input_name, input_desc, desc_visible])
                    
                    with gr.Column():
                        for i, output_value in enumerate(output_values):
                            with gr.Group():
                                with gr.Row():
                                    output_name = gr.Textbox(
                                        placeholder=output_value,
                                        key=f"output-name-{i}",
                                        show_label=False,
                                        label=f"Output {i+1} Name",
                                        info="Specify the name of this output field.",
                                        scale=9
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                output_desc = gr.Textbox(
                                    placeholder="Description (optional)",
                                    key=f"output-desc-{i}",
                                    show_label=False,
                                    label=f"Output {i+1} Description",
                                    info="Optionally provide a description for this output field.",
                                    visible=False,
                                )
                                desc_visible = gr.State(False)
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, output_desc]
                                )
                                outputs.extend([output_name, output_desc, desc_visible])

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
                        value="BootstrapFewShot",
                        info="Choose optimization strategy: None (no optimization), BootstrapFewShot (small datasets, ~10 examples) uses few-shot learning; BootstrapFewShotWithRandomSearch (medium, ~50) adds randomized search; MIPRO, MIPROv2, and COPRO (large, 300+) also optimize the prompt instructions."
                    )
                    with gr.Column():
                        metric_type = gr.Radio(
                            ["Exact Match", "Cosine Similarity", "LLM-as-a-Judge"],
                            label="Metric",
                            value="Exact Match",
                            info="Choose how to evaluate your program's performance. Exact Match is suitable for tasks with clear correct answers, while LLM-as-a-Judge is better for open-ended or subjective tasks. Cosine Similarity can be used for fuzzier matches tasks where the output needs to be similar to the correct answer."
                        )
                        judge_prompt = gr.Dropdown(
                            choices=[],
                            label="Judge Prompt",
                            visible=False,
                            info="Select the prompt to use as the judge for evaluation."
                        )

                def update_judge_prompt_visibility(metric):
                    if metric == "LLM-as-a-Judge":
                        prompts = get_available_prompts()
                        return gr.update(visible=True, choices=[f"{p['id']} - {p['signature']} (Score: {p['eval_score']})" for p in prompts])
                    else:
                        return gr.update(visible=False, choices=[])

                metric_type.change(
                    update_judge_prompt_visibility,
                    inputs=[metric_type],
                    outputs=[judge_prompt]
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
                    input_fields = []
                    input_descs = []
                    output_fields = []
                    output_descs = []
                    
                    for i in range(0, len(inputs), 3):
                        if data[inputs[i]].strip():
                            input_fields.append(data[inputs[i]])
                            if data[inputs[i+1]].strip():
                                input_descs.append(data[inputs[i+1]])
                    
                    for i in range(0, len(outputs), 3):
                        if data[outputs[i]].strip():
                            output_fields.append(data[outputs[i]])
                            if data[outputs[i+1]].strip():
                                output_descs.append(data[outputs[i+1]])

                    # Get the judge prompt ID if LLM-as-a-Judge is selected
                    judge_prompt_id = None
                    if data[metric_type] == "LLM-as-a-Judge":
                        judge_prompt_id = data[judge_prompt].split(' - ')[0]

                    usage_instructions, optimized_prompt = compile_program(
                        input_fields,
                        output_fields,
                        data[dspy_module],
                        data[llm_model], 
                        data[teacher_model],
                        data[example_data],
                        data[optimizer],
                        data[instructions],
                        data[metric_type],
                        judge_prompt_id,
                        input_descs,
                        output_descs
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
                    inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, upload_csv_btn, optimizer, instructions, metric_type, judge_prompt]),
                    outputs=[signature, evaluation_score, optimized_prompt, usage_instructions]
                )

        with gr.TabItem("View Prompts"):
            gr.Markdown("# View Prompts")
            prompts = list_prompts()

            selected_prompt = gr.State(None)
            
            # Extract unique signatures for the dropdown
            unique_signatures = sorted(set(p["Signature"] for p in prompts))

            # Add filter and sort functionality in one line
            with gr.Row():
                filter_signature = gr.Dropdown(label="Filter by Signature", choices=["All"] + unique_signatures, value="All", scale=2)
                sort_by = gr.Radio(["Run Date", "Evaluation Score"], label="Sort by", value="Run Date", scale=1)
                sort_order = gr.Radio(["Descending", "Ascending"], label="Sort Order", value="Descending", scale=1)

            @gr.render(inputs=[selected_prompt])
            def render_prompt_details(selected_prompt):
                if selected_prompt is not None:
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("## Prompt Details")
                            with gr.Group():
                                with gr.Column(elem_classes="prompt-details-full"):
                                    gr.Markdown(f"**ID:** {selected_prompt['ID']}")
                                    gr.Markdown(f"**Signature:** {selected_prompt['Signature']}")
                                    gr.Markdown(f"**Eval Score:** {selected_prompt['Eval Score']}")
                                    details = json.loads(selected_prompt["Details"])
                                    formatted_details = "\n".join([f"**{key}:** {value}" for key, value in details.items()])
                                    gr.Markdown(formatted_details)
            
            close_details_btn = gr.Button("Close Details", elem_classes="close-details-btn", size="sm", visible=False)
            close_details_btn.click(lambda: (None, gr.update(visible=False)), outputs=[selected_prompt, close_details_btn])
            
            @gr.render(inputs=[filter_signature, sort_by, sort_order])
            def render_prompts(filter_signature, sort_by, sort_order):
                signature = filter_signature
                order = sort_order

                if signature and signature != "All":
                    filtered_prompts = [p for p in prompts if p["Signature"] == signature]
                else:
                    filtered_prompts = prompts
                
                if sort_by == "Evaluation Score":
                    key_func = lambda x: float(x["Eval Score"])
                else:  # Run Date
                    key_func = lambda x: x["ID"]  # Use the entire ID for sorting
                
                sorted_prompts = sorted(filtered_prompts, key=key_func, reverse=(order == "Descending"))
                
                for i in range(0, len(sorted_prompts), 3):
                    with gr.Row():
                        for j in range(3):
                            if i + j < len(sorted_prompts):
                                prompt = sorted_prompts[i + j]
                                with gr.Column():
                                    with gr.Group(elem_classes="prompt-card"):
                                        with gr.Column(elem_classes="prompt-details"):
                                            gr.Markdown(f"**ID:** {prompt['ID']}")
                                            gr.Markdown(f"**Signature:** {prompt['Signature']}")
                                            gr.Markdown(f"**Eval Score:** {prompt['Eval Score']}")
                                        view_details_btn = gr.Button("View Details", elem_classes="view-details-btn", size="sm")
                                    
                                    view_details_btn.click(lambda: (prompt, gr.update(visible=True)), outputs=[selected_prompt, close_details_btn])

                

# Launch the interface
iface.launch()