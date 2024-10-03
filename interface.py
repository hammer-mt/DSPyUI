import gradio as gr
import pandas as pd
import json
import os
import glob

from core import compile_program, list_prompts, export_to_csv


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

    # Compile Program Tab
    with gr.Tabs():
        with gr.TabItem("Compile Program"):

            with gr.Row():
                with gr.Column():
                    gr.Markdown("# DSPyUI: a Gradio user interface for DSPy")
                    gr.Markdown("Compile a DSPy program by specifying your settings and providing example data.")

                with gr.Column():
                    gr.Markdown("### Demo Examples:")
                    with gr.Row():  
                        example1 = gr.Button("Judging Jokes")
                        example2 = gr.Button("Telling Jokes")
                        example3 = gr.Button("Rewriting Jokes")
            
            # Task Instructions
            with gr.Row():
                with gr.Column(scale=4):
                    instructions = gr.Textbox(
                        label="Task Instructions",
                        lines=3,
                        placeholder="Enter detailed task instructions here.",
                        info="Provide clear and comprehensive instructions for the task. This will guide the DSPy program in understanding the specific requirements and expected outcomes.",
                        interactive=True  # Add this line to ensure the textbox is editable
                    )

            input_values = gr.State([])
            output_values = gr.State([])
            file_data = gr.State(None)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Inputs")
                    with gr.Row():
                        add_input_btn = gr.Button("Add Input Field")
                        remove_input_btn = gr.Button("Remove Last Input", interactive=False)
                with gr.Column():
                    gr.Markdown("### Outputs")
                    with gr.Row():  
                        add_output_btn = gr.Button("Add Output Field")
                        remove_output_btn = gr.Button("Remove Last Output", interactive=False)

            def add_field(values):
                new_values = values + [("", "")]
                return new_values, gr.update(interactive=True)

            def remove_last_field(values):
                new_values = values[:-1] if values else values
                return new_values, gr.update(interactive=bool(new_values))

            add_input_btn.click(
                add_field,
                inputs=input_values,
                outputs=[input_values, remove_input_btn]
            )

            remove_input_btn.click(
                remove_last_field,
                inputs=input_values,
                outputs=[input_values, remove_input_btn]
            )

            add_output_btn.click(
                add_field,
                inputs=output_values,
                outputs=[output_values, remove_output_btn]
            )

            remove_output_btn.click(
                remove_last_field,
                inputs=output_values,
                outputs=[output_values, remove_output_btn]
            )

            def load_csv(filename):
                try:
                    df = pd.read_csv(f"example_data/{filename}")
                    return df
                except Exception as e:
                    print(f"Error loading CSV: {e}")
                    return None

            @gr.render(inputs=[input_values, output_values, file_data])
            def render_variables(input_values, output_values, file_data):
                inputs = []
                outputs = []
                with gr.Row():
                    with gr.Column():
                        for i, input_value in enumerate(input_values):
                            name, desc = input_value
                            with gr.Group():
                                with gr.Row():
                                    input_name = gr.Textbox(
                                        placeholder=f"Input{i+1}",
                                        value=name if name else None,
                                        key=f"input-name-{i}",
                                        show_label=False,
                                        label=f"Input {i+1} Name",
                                        info="Specify the name of this input field.",
                                        interactive=True,
                                        scale=9
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                input_desc = gr.Textbox(
                                    value=desc if desc else None,
                                    placeholder=desc if desc else "Description (optional)",
                                    key=f"input-desc-{i}",
                                    show_label=False,
                                    label=f"Input {i+1} Description",
                                    info="Optionally provide a description for this input field.",
                                    interactive=True,
                                    visible=False
                                )
                                desc_visible = gr.State(lambda: bool(desc))
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, input_desc]
                                )
                                inputs.extend([input_name, input_desc, desc_visible])
                    
                    with gr.Column():
                        for i, output_value in enumerate(output_values):
                            name, desc = output_value
                            with gr.Group():
                                with gr.Row():
                                    output_name = gr.Textbox(
                                        placeholder=f"Output{i+1}",
                                        value=name if name else None,
                                        key=f"output-name-{i}",
                                        show_label=False,
                                        label=f"Output {i+1} Name",
                                        info="Specify the name of this output field.",
                                        scale=9,
                                        interactive=True,
                                    )
                                    expand_btn = gr.Button("▼", size="sm", scale=1, elem_classes="expand-button")
                                output_desc = gr.Textbox(
                                    value=desc if desc else None,
                                    placeholder=desc if desc else "Description (optional)",
                                    key=f"output-desc-{i}",
                                    show_label=False,
                                    label=f"Output {i+1} Description",
                                    info="Optionally provide a description for this output field.",
                                    visible=False,
                                    interactive=True,
                                )
                                desc_visible = gr.State(lambda: bool(desc))
                                expand_btn.click(
                                    lambda v: (not v, gr.update(visible=not v)),
                                    inputs=[desc_visible],
                                    outputs=[desc_visible, output_desc]
                                )
                                outputs.extend([output_name, output_desc, desc_visible])

                    def update_judge_prompt_visibility(metric, *args):
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_values):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)

                        if metric == "LLM-as-a-Judge":
                            
                            prompts = list_prompts(output_filter=input_fields + output_fields)

                            return gr.update(visible=True, choices=[f"{p['ID']} - {p['Signature']} (Score: {p['Eval Score']})" for p in prompts])
                        else:
                            return gr.update(visible=False, choices=[])

                    metric_type.change(
                        update_judge_prompt_visibility,
                        inputs=[metric_type] + inputs + outputs,
                        outputs=[judge_prompt]
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

                        hint = data[hint_textbox] if data[dspy_module] == "ChainOfThoughtWithHint" else None
                        
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
                            output_descs,
                            hint  # Add the hint parameter
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
                            "input_descriptions": input_descs,
                            "output_fields": output_fields,
                            "output_descriptions": output_descs,
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
                    
                gr.Markdown("### Data")
                with gr.Column():
                    with gr.Row():
                        enter_manually_btn = gr.Button("Enter manually", interactive=len(input_values) > 0 and len(output_values) > 0)
                        
                        upload_csv_btn = gr.UploadButton("Upload CSV", file_types=[".csv"], interactive=len(input_values) > 0 and len(output_values) > 0)

                    headers = [input_value[0] for input_value in input_values] + [output_value[0] for output_value in output_values]
                        
                    example_data = gr.Dataframe(
                        headers=headers,
                        datatype=["str"] * (len(input_values) + len(output_values)),
                        interactive=True,
                        row_count=1,
                        col_count=(len(input_values) + len(output_values), "fixed"),
                        visible=file_data is not None,  # Only visible if file_data is not None
                        label="Example Data",
                        value=file_data if file_data is not None else pd.DataFrame(columns=headers)
                    )
                    export_csv_btn = gr.Button("Export to CSV", interactive=file_data is not None and len(input_values) > 0 and len(output_values) > 0)
                    csv_download = gr.File(label="Download CSV", visible=False)
                    error_message = gr.Markdown()
                    
                    def show_dataframe(*args):
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        input_names = [name for name, _ in input_values]
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_names):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)

                        headers = input_fields + output_fields
                        
                        # Create a new dataframe with the correct headers
                        new_df = pd.DataFrame(columns=headers)
                        
                        return gr.update(visible=True, value=new_df), gr.update(visible=True), gr.update(visible=True)

                    enter_manually_btn.click(
                        show_dataframe,
                        inputs=inputs + outputs,
                        outputs=[example_data, export_csv_btn, compile_button]
                    )
                    upload_csv_btn.upload(
                        process_csv,
                        inputs=[upload_csv_btn] + inputs + outputs,
                        outputs=[example_data, example_data, compile_button, error_message]
                    )

                    export_csv_btn.click(
                        export_to_csv,
                        inputs=[example_data],
                        outputs=[csv_download]
                    ).then(
                        lambda: gr.update(visible=True),
                        outputs=[csv_download]
                    )

                compile_button.click(
                    compile,
                    inputs=set(inputs + outputs + [llm_model, teacher_model, dspy_module, example_data, upload_csv_btn, optimizer, instructions, metric_type, judge_prompt, hint_textbox]),
                    outputs=[signature, evaluation_score, optimized_prompt, usage_instructions]
                )

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
                with gr.Column():
                    dspy_module = gr.Dropdown(
                        ["Predict", "ChainOfThought", "ChainOfThoughtWithHint"],
                        label="Module",
                        value="Predict",
                        info="Choose the DSPy module that best fits your task. Predict is for simple tasks, ChainOfThought for complex reasoning, and ChainOfThoughtWithHint for guided reasoning."
                    )
                    hint_textbox = gr.Textbox(
                        label="Hint",
                        lines=2,
                        placeholder="Enter a hint for the Chain of Thought with Hint module.",
                        visible=False
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

            compile_button = gr.Button("Compile Program", visible=False, variant="primary")
            with gr.Column() as compilation_results:
                gr.Markdown("### Results")
                signature = gr.Textbox(label="Signature", interactive=False, info="The compiled signature of your DSPy program, showing inputs and outputs.")
                evaluation_score = gr.Number(label="Evaluation Score", info="The evaluation score of your compiled DSPy program.")
                optimized_prompt = gr.Textbox(label="Optimized Prompt", info="The optimized prompt generated by the DSPy compiler for your program.")
                usage_instructions = gr.Textbox(label="Usage Instructions", info="Instructions on how to use your compiled DSPy program.")

            def process_csv(file, *args):
                if file is not None:
                    try:
                        df = pd.read_csv(file.name)
                        # Correctly assign input and output fields based on the actual arguments
                        input_fields = []
                        output_fields = []
                        filtered_args = [args[i] for i in range(0, len(args), 3)]  # Filter out descriptions and visibility
                        for arg in filtered_args:
                            if arg and isinstance(arg, str) and arg.strip():
                                if len(input_fields) < len(input_values):
                                    input_fields.append(arg)
                                elif len(output_fields) < len(output_values):
                                    output_fields.append(arg)
                        expected_headers = input_fields + output_fields
                        
                        if list(df.columns) != expected_headers:
                            return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: CSV headers do not match expected format. Expected: {expected_headers}, Got: {list(df.columns)}")
                        return df, gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
                    except Exception as e:
                        return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, value=f"Error: {str(e)}")
                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

            # Function to show/hide the hint textbox based on the selected module
            def update_hint_visibility(module):
                return gr.update(visible=module == "ChainOfThoughtWithHint")

            # Connect the visibility update function to the module dropdown
            dspy_module.change(update_hint_visibility, inputs=[dspy_module], outputs=[hint_textbox])

            
            def disable_example_buttons():
                return gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)

            example1.click(
                lambda _: (
                    gr.update(value="Rate whether a joke is funny"),
                    gr.update(value="BootstrapFewShotWithRandomSearch"),
                    gr.update(value="Exact Match"),
                    gr.update(value="gpt-4o-mini"),
                    gr.update(value="gpt-4o"),
                    gr.update(value="ChainOfThought"),
                    [("joke", "The joke to be rated"), ("topic", "The topic of the joke")],
                    [("funny", "Whether the joke is funny or not, 1 or 0.")],
                    *disable_example_buttons(),
                    load_csv("rating_jokes.csv")
                ),
                inputs=[gr.State(None)],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data]
            )

            example2.click(
                lambda _: (
                    gr.update(value="Tell me a funny joke"),
                    gr.update(value="MIPROv2"),
                    gr.update(value="LLM-as-a-Judge"),
                    gr.update(value="gpt-4o-mini"),
                    gr.update(value="gpt-4o"),
                    gr.update(value="Predict"),
                    [("topic", "The topic of the joke")],
                    [("joke", "The funny joke")],
                    *disable_example_buttons(),
                    load_csv("telling_jokes.csv")
                ),
                inputs=[gr.State(None)],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data]
            )

            example3.click(
                lambda _: (
                    gr.update(value="Rewrite in a comedian's style"),
                    gr.update(value="BootstrapFewShot"),
                    gr.update(value="Cosine Similarity"),
                    gr.update(value="claude-3-haiku-20240307"),
                    gr.update(value="claude-3-sonnet-20240229"),
                    gr.update(value="Predict"),
                    [("joke", "The joke to be rewritten"), ("comedian", "The comedian the joke should be rewritten in the style of")],
                    [("rewritten_joke", "The rewritten joke")],
                    *disable_example_buttons(),
                    load_csv("rewriting_jokes.csv")
                ),
                inputs=[gr.State(None)],
                outputs=[instructions, optimizer, metric_type, llm_model, teacher_model, dspy_module, input_values, output_values, example1, example2, example3, file_data]
            )

            

        with gr.TabItem("View Prompts"):
            
            prompts = list_prompts()

            selected_prompt = gr.State(None)
            
            # Extract unique signatures for the dropdown
            unique_signatures = sorted(set(p["Signature"] for p in prompts))

            close_details_btn = gr.Button("Close Details", elem_classes="close-details-btn", size="sm", visible=False)
            close_details_btn.click(lambda: (None, gr.update(visible=False)), outputs=[selected_prompt, close_details_btn])
            

            @gr.render(inputs=[selected_prompt])
            def render_prompt_details(selected_prompt):
                if selected_prompt is not None:
                    with gr.Row():
                        with gr.Column():
                            details = json.loads(selected_prompt["Details"])
                            gr.Markdown(f"## {details['human_readable_id']}")
                            with gr.Group():
                                with gr.Column(elem_classes="prompt-details-full"):
                                    gr.Number(value=float(selected_prompt['Eval Score']), label="Evaluation Score", interactive=False)
                                    
                                    with gr.Row():
                                        gr.Dropdown(choices=details['input_fields'], value=details['input_fields'], label="Input Fields", interactive=False, multiselect=True, info=", ".join(details.get('input_descs', [])))
                                        gr.Dropdown(choices=details['output_fields'], value=details['output_fields'], label="Output Fields", interactive=False, multiselect=True, info=", ".join(details.get('output_descs', [])))
                                    
                                    with gr.Row():
                                        gr.Dropdown(choices=[details['dspy_module']], value=details['dspy_module'], label="Module", interactive=False)
                                        gr.Dropdown(choices=[details['llm_model']], value=details['llm_model'], label="Model", interactive=False)
                                        gr.Dropdown(choices=[details['teacher_model']], value=details['teacher_model'], label="Teacher Model", interactive=False)
                                        gr.Dropdown(choices=[details['optimizer']], value=details['optimizer'], label="Optimizer", interactive=False)
                                    
                                    gr.Textbox(value=details['instructions'], label="Instructions", interactive=False)
                                    
                                    gr.Textbox(value=details['optimized_prompt'], label="Optimized Prompt", interactive=False)
                                    
                                    for key, value in details.items():
                                        if key not in ['signature', 'evaluation_score', 'usage_instructions', 'input_fields', 'output_fields', 'dspy_module', 'llm_model', 'teacher_model', 'optimizer', 'instructions', 'optimized_prompt', 'human_readable_id']:
                                            if isinstance(value, list):
                                                gr.Dropdown(choices=value, value=value, label=key.replace('_', ' ').title(), interactive=False, multiselect=True)
                                            elif isinstance(value, bool):
                                                gr.Checkbox(value=value, label=key.replace('_', ' ').title(), interactive=False)
                                            elif isinstance(value, (int, float)):
                                                gr.Number(value=value, label=key.replace('_', ' ').title(), interactive=False)
                                            else:
                                                gr.Textbox(value=str(value), label=key.replace('_', ' ').title(), interactive=False)
                        

            gr.Markdown("# View Prompts")
            
            # Add filter and sort functionality in one line
            with gr.Row():
                filter_signature = gr.Dropdown(label="Filter by Signature", choices=["All"] + unique_signatures, value="All", scale=2)
                sort_by = gr.Radio(["Run Date", "Evaluation Score"], label="Sort by", value="Run Date", scale=1)
                sort_order = gr.Radio(["Descending", "Ascending"], label="Sort Order", value="Descending", scale=1)

            @gr.render(inputs=[filter_signature, sort_by, sort_order])
            def render_prompts(filter_signature, sort_by, sort_order):
                if filter_signature and filter_signature != "All":
                    filtered_prompts = list_prompts(signature_filter=filter_signature)
                else:
                    filtered_prompts = prompts
                
                if sort_by == "Evaluation Score":
                    key_func = lambda x: float(x["Eval Score"])
                else:  # Run Date
                    key_func = lambda x: x["ID"]  # Use the entire ID for sorting
                
                sorted_prompts = sorted(filtered_prompts, key=key_func, reverse=(sort_order == "Descending"))
                
                prompt_components = []
                
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
                                    
                                    prompt_components.append((prompt, view_details_btn))
                
                for prompt, btn in prompt_components:
                    btn.click(
                        lambda p=prompt: (p, gr.update(visible=True)),
                        outputs=[selected_prompt, close_details_btn]
                    )

# Launch the interface
iface.launch()