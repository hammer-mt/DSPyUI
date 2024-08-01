import gradio as gr

with gr.Blocks() as demo:
    input_count = gr.State(1)
    add_input_btn = gr.Button("Add Input")

    add_input_btn.click(lambda count: count + 1, input_count, input_count)

    @gr.render(inputs=input_count)
    def render_tracks(count):
        inputs = []
        with gr.Row():
            for i in range(count):
                with gr.Column(variant="panel", min_width=200):
                    
                    input = gr.Textbox(placeholder="Input", key=f"input-{i}", show_label=True)
                    inputs.append(input)


            def merge(data):
                output = ""
                for input in inputs:
                    input_val = data[input]
                    output += input_val + ", "

                return output

            submit_btn.click(merge, set(inputs), output_inputs)

    submit_btn = gr.Button("Print Inputs")
    output_inputs = gr.Textbox(label="Output", interactive=False)

demo.launch()
