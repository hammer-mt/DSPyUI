import gradio as gr

with gr.Blocks() as demo:
    track_count = gr.State(1)
    add_track_btn = gr.Button("Add Track")

    add_track_btn.click(lambda count: count + 1, track_count, track_count)

    @gr.render(inputs=track_count)
    def render_tracks(count):
        names = []
        with gr.Row():
            for i in range(count):
                with gr.Column(variant="panel", min_width=200):
                    
                    track_name = gr.Textbox(placeholder="Track Name", key=f"name-{i}", show_label=True)
                    names.append(track_name)


            def merge(data):
                output = ""
                for name in names:
                    name_val = data[name]
                    output += name_val + ", "

                return output

            merge_btn.click(merge, set(names), output_audio)

    merge_btn = gr.Button("Merge Tracks")
    output_audio = gr.Textbox(label="Output", interactive=False)

demo.launch()
