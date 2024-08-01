import gradio as gr

with gr.Blocks() as demo:
    input_count = gr.State(1)
    output_count = gr.State(1)

    @gr.render(inputs=[input_count, output_count])
    def render_tracks(input_count, output_count):
        inputs = []
        outputs = []
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Inputs")
                for i in range(input_count):
                    with gr.Group():
                        input = gr.Textbox(placeholder="Input{i}", key=f"input-{i}", show_label=False)
                        inputs.append(input)
            
            with gr.Column():
                gr.Markdown("### Outputs")
                for i in range(output_count):
                    with gr.Group():
                        output = gr.Textbox(placeholder="Output{i}", key=f"output-{i}", show_label=False)
                        outputs.append(output)

        def compile(data):
            input_text = ", ".join(data[input] for input in inputs if data[input])
            output_text = ", ".join(data[output] for output in outputs if data[output])
            result = f"{input_text} -> {output_text}"
            return result

        submit_btn.click(compile, set(inputs + outputs), final_result)

    with gr.Row():
        with gr.Column():
            add_input_btn = gr.Button("Add Input")
            add_input_btn.click(lambda count: count + 1, input_count, input_count)
        with gr.Column():
            add_output_btn = gr.Button("Add Output")
            add_output_btn.click(lambda count: count + 1, output_count, output_count)

    submit_btn = gr.Button("Print Signature")
    final_result = gr.Textbox(label="Final Result", interactive=False)

demo.launch()
