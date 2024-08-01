import gradio as gr

with gr.Blocks() as demo:
    input_count = gr.State(1)
    output_count = gr.State(1)
    add_input_btn = gr.Button("Add Input")
    add_output_btn = gr.Button("Add Output")

    add_input_btn.click(lambda count: count + 1, input_count, input_count)
    add_output_btn.click(lambda count: count + 1, output_count, output_count)

    @gr.render(inputs=[input_count, output_count])
    def render_tracks(input_count, output_count):
        inputs = []
        outputs = []
        with gr.Row():
            with gr.Column():
                for i in range(input_count):
                    with gr.Group():
                        input = gr.Textbox(placeholder="Input", key=f"input-{i}", show_label=False)
                        inputs.append(input)
            
            with gr.Column():
                for i in range(output_count):
                    with gr.Group():
                        output = gr.Textbox(placeholder="Output", key=f"output-{i}", show_label=False)
                        outputs.append(output)

        def compile(data):
            input_text = ""
            for input in inputs:
                input_val = data[input]
                input_text += input_val + ", "
            
            output_text = ""
            for output in outputs:
                output_val = data[output]
                output_text += output_val + ", "

            return input_text, output_text

        submit_btn.click(compile, set(inputs + outputs), [input_result, output_result])

    submit_btn = gr.Button("Print Inputs and Outputs")
    input_result = gr.Textbox(label="Input Result", interactive=False)
    output_result = gr.Textbox(label="Output Result", interactive=False)

demo.launch()
