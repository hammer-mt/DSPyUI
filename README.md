# DSPyUI

A Gradio user interface for DSPy - making DSPy accessible to non-technical users through a visual web interface.

## Quick Start

```bash
bash webui.sh
```

## Requirements

- Python 3.10 or higher (required for DSPy 3.0)
- API keys for LLM providers (OpenAI, Anthropic, Groq, and/or Google)

## Setup

1. Clone the repository
2. Create a `.env` file with your API keys (see `.env.example`)
3. Run `bash webui.sh` to start the application

The script will automatically:
- Create a virtual environment
- Install dependencies (Gradio 5.x, DSPy 3.0, and LLM clients)
- Load environment variables from `.env`
- Launch the Gradio interface

## Manual Installation

```bash
python3 -m venv dspyui_env
source dspyui_env/bin/activate
pip install -r requirements.txt
gradio interface.py
```

## Features

- Visual interface for DSPy program compilation
- Support for multiple LLM providers (OpenAI, Anthropic, Groq, Google)
- **Local LLM support** (LM Studio, Ollama, llama.cpp)
- Built-in DSPy modules (Predict, ChainOfThought, ChainOfThoughtWithHint, **ProgramOfThought**)
- Built-in optimizers (BootstrapFewShot, MIPRO, COPRO, **LabeledFewShot**, and more)
- Evaluation metrics (Exact Match, Cosine Similarity, LLM-as-a-Judge)
- **Evaluation in generate interface** - see scores when testing predictions
- Example datasets and pre-compiled programs
- Export and reuse compiled programs
- Load existing prompts back into the compiler

## Technology Stack

- **UI Framework**: Gradio 5.x
- **DSPy Version**: 3.0.4b1 (vendored in `./dspy/`)
- **Python**: 3.10+

## Supported Modules

- **Predict** - Simple prediction for straightforward tasks
- **ChainOfThought** - Step-by-step reasoning for complex tasks
- **ChainOfThoughtWithHint** - Guided reasoning with custom hints
- **ProgramOfThought** - Code execution for computational tasks (requires [deno](https://docs.deno.com/runtime/getting_started/installation/))

## Supported Optimizers

- **BootstrapFewShot** - Few-shot learning for small datasets (~10 examples)
- **BootstrapFewShotWithRandomSearch** - Few-shot with random search for medium datasets (~50 examples)
- **MIPROv2** - Instruction optimization for large datasets (300+ examples)
- **COPRO** - Prompt optimization for large datasets (300+ examples)
- **LabeledFewShot** - Simple labeled examples without optimization (any dataset size)

## Local LLM Support

DSPyUI supports running local LLMs through OpenAI-compatible endpoints:

1. **LM Studio**: Start LM Studio server (default: http://127.0.0.1:1234/v1)
2. **Ollama**: Use with OpenAI compatibility enabled
3. **llama.cpp**: Run server with `--api-key` flag

To use:
- In the model dropdown, select a model name starting with `local:`
- Enter your local server's base URL (or use the default for LM Studio)
- Compile and run as normal - data stays completely private!

<img width="1561" alt="Screenshot 2025-05-23 at 09 53 48" src="https://github.com/user-attachments/assets/df95d7ee-c605-47cc-a389-19cdd67f7a02" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 54 33" src="https://github.com/user-attachments/assets/e3cea6f3-68eb-4c48-bb6d-c5ef01eba827" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 53 58" src="https://github.com/user-attachments/assets/ea9d73bb-027e-4f3f-ae0d-b27fedaaf61d" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 54 08" src="https://github.com/user-attachments/assets/f34858ca-14d8-4091-aa78-05ff8150defe" />
