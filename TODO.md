## Bugs
- AttributeError: 'NoneType' object has no attribute 'headers'
- fn = session_state.blocks_config.fns[body.fn_index]         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: 12

## Backlog
- Upgrade to DSPy 3.0 and latest version of gradio
- Build some way to chain the modules together in agentic workflows (like the ComfyUI canvas approach or N8N)
- add a place to put API keys then push to gradio spaces: https://x.com/victormustar/status/1844844179962450331?t=o-wTXyD8CsXllPfRGN3wMg&s=03
- add cost tracking and prediction of cost to the interface
- support all the various optimizers and modules in the interface: 
    - ProgramOfThought, ReAct, MultiChainComparison, majority
    - BootstrapFineTune, Ensemble, LabeledFewShot
- add RAG support: Retrieve, Retrieval Model Clients
- multimodal support with Image
- local running of LLM with olama, llama.cpp or lm studio
- ability to load an existing prompt back into the interface to re-run
- consolidate the output of the program to a single file
- run the evaluation of the program in the generate interface too