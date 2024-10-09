## Bugs
- AttributeError: 'NoneType' object has no attribute 'headers'

## Backlog
- update for the latest DSPy version / experimental features
- add cost tracking and prediction of cost to the interface
- support all the various optimizers and modules in the interface: 
    - ProgramOfThought, ReAct, MultiChainComparison, majority
    - BootstrapFineTune, Ensemble, LabeledFewShot
- add RAG support: Retrieve, Retrieval Model Clients
- local running of LLM with olama, llama.cpp or lm studio
- ability to load an existing prompt back into the interface to re-run
- consolidate the output of the program to a single file
- run the evaluation of the program in the generate interface too