## Bugs
None currently! Previous bugs were fixed:
- ✅ AttributeError: 'NoneType' object has no attribute 'headers' (fixed with Gradio 5 upgrade)
- ✅ KeyError: 12 in blocks_config.fns (fixed with Gradio 5 upgrade)

## Backlog
- ✅ Upgrade to DSPy 3.0 and latest version of gradio (COMPLETED - DSPy 3.0.3, Gradio 5.48.0)
- Build some way to chain the modules together in agentic workflows (like the ComfyUI canvas approach or N8N)
- add a place to put API keys then push to gradio spaces: https://x.com/victormustar/status/1844844179962450331?t=o-wTXyD8CsXllPfRGN3wMg&s=03
- ✅ add cost tracking and prediction of cost to the interface (COMPLETED - actual cost tracking and display in UI)
- support all the various optimizers and modules in the interface:
    - ✅ ProgramOfThought (COMPLETED - requires deno for code execution)
    - ✅ LabeledFewShot (COMPLETED)
    - ✅ BootstrapFineTune (COMPLETED)
    - ❌ ReAct (requires tool definition UI - deferred)
    - ❌ MultiChainComparison (requires completions architecture - deferred)
    - ❌ Ensemble (requires multi-program compilation - deferred)
    - ℹ️ majority (not a module, it's a function used by Ensemble)
- add RAG support: Retrieve, Retrieval Model Clients
- multimodal support with Image
- ✅ local running of LLM with ollama, llama.cpp or lm studio (COMPLETED - OpenAI-compatible endpoints)
- ✅ ability to load an existing prompt back into the interface to re-run (COMPLETED)
- consolidate the output of the program to a single file
- ✅ run the evaluation of the program in the generate interface too (COMPLETED - displays score with same metrics as compilation)