"""
Launcher for DSPyUI, kept at the top of the repository so that
`bash webui.sh`, `gradio interface.py` and `python interface.py` keep working.

The app itself lives in the dspyui/ package:

    dspyui/interface.py   the Gradio web form (this is the UI code)
    dspyui/core.py        the DSPy logic the book walks through in Chapter 10:
                          create_custom_signature(), MODEL_OPTIONS, create_module(),
                          load_and_split(), compile_program(), run_inference()
"""

# The app is built in dspyui/interface.py (`with gr.Blocks(css=custom_css) as demo:`)
from dspyui.interface import demo

if __name__ == "__main__":
    demo.launch()
