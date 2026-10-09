"""
Launcher for DSPyUI, kept at the top of the repository so that
`bash webui.sh`, `gradio interface.py` and `python interface.py` keep working.

The app itself lives in the dspyui/ package:

    dspyui/interface.py   the Gradio web form (this is the UI code)
    dspyui/core.py        the DSPy logic the book walks through in Chapter 10:
                          create_custom_signature(), MODEL_OPTIONS, create_module(),
                          load_and_split(), compile_program(), run_inference()
"""

import sys

# The app is built in dspyui/interface.py (`with gr.Blocks(css=custom_css) as demo:`).
#
# `gradio interface.py` (which webui.sh uses) runs in reload mode: Gradio resets
# its component counter and runs this file a second time, expecting that to
# build the interface again. Forgetting the module first makes the import below
# do exactly that; without it the counter stays at zero and the parts of the
# page that are drawn on demand (the Data section) fail to render.
sys.modules.pop("dspyui.interface", None)
from dspyui.interface import demo

if __name__ == "__main__":
    demo.launch()
