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
- Built-in optimizers (BootstrapFewShot, BootstrapFewShotWithRandomSearch, MIPROv2, COPRO, BootstrapFinetune, **LabeledFewShot**)
- Evaluation metrics (Exact Match, Cosine Similarity, LLM-as-a-Judge)
- **Evaluation in generate interface** - see scores when testing predictions
- **Consolidated .dspyui file format** - all program data in a single file
- Example datasets and pre-compiled programs
- Export and reuse compiled programs
- Load existing prompts back into the compiler

## Technology Stack

- **UI Framework**: Gradio 5.48.0
- **DSPy Version**: 3.0.3 (pip-installed)
- **Python**: 3.10+
- **Type Safety**: mypy with comprehensive type hints
- **Data Validation**: Pydantic models for configuration
- **Testing**: Playwright for end-to-end tests, pytest for unit tests

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
- **BootstrapFinetune** - Distill prompts into fine-tuned model weights
- **LabeledFewShot** - Simple labeled examples without optimization (any dataset size)

## Evaluation Features

DSPyUI provides comprehensive evaluation capabilities:

### During Compilation
- Choose from three evaluation metrics:
  - **Exact Match**: Compare outputs for exact equality
  - **Cosine Similarity**: Semantic similarity using embeddings
  - **LLM-as-a-Judge**: Use a separate DSPy program as an evaluator
- Automatic train/dev split (80/20) for validation
- Baseline and optimized scores displayed

### During Generation
- When testing individual predictions, evaluation scores are automatically displayed
- Works with the same metrics as compilation
- Requires gold/expected output data to be present
- Helpful for debugging and quality assurance

## Local LLM Support

DSPyUI supports running local LLMs through OpenAI-compatible endpoints:

1. **LM Studio**: Start LM Studio server (default: http://127.0.0.1:1234/v1)
2. **Ollama**: Use with OpenAI compatibility enabled
3. **llama.cpp**: Run server with `--api-key` flag

To use:
- In the model dropdown, select a model name starting with `local:`
- Enter your local server's base URL (or use the default for LM Studio)
- Compile and run as normal - data stays completely private!

## Consolidated File Format (.dspyui)

DSPyUI automatically saves compiled programs in two formats:
1. **Legacy format**: Three separate files (dataset.csv, program.json, prompt.json)
2. **Consolidated format**: Single `.dspyui` file containing all program data

### Benefits of .dspyui Format
- **Single file** for easy sharing and backup
- **Complete program package** - includes configuration, compiled program, and training data
- **Version controlled** - includes DSPy and DSPyUI version info
- **Future-proof** - foundation for workflow chaining and other advanced features

### Export Existing Programs
Click the "💾 Export .dspyui" button in the View Prompts tab to convert any existing program to the consolidated format. Files are saved to `consolidated_programs/`.

## Code Quality & Testing

DSPyUI maintains high code quality standards:

### Type Safety
- Comprehensive type hints on all core functions
- mypy configuration for static type checking
- Pydantic models for runtime data validation
- Clear function signatures for better IDE support

### Testing Infrastructure
- Automated Gradio test runner (`gradio_test_runner.py`)
- Playwright end-to-end tests (15/20 passing)
- Mock data generation for testing
- Comprehensive logging and error tracking

Run tests:
```bash
source dspyui_env/bin/activate
pytest tests/ -v                    # Run Playwright tests
python gradio_test_runner.py       # Run with Gradio server
mypy models.py                     # Type checking
```

<img width="1561" alt="Screenshot 2025-05-23 at 09 53 48" src="https://github.com/user-attachments/assets/df95d7ee-c605-47cc-a389-19cdd67f7a02" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 54 33" src="https://github.com/user-attachments/assets/e3cea6f3-68eb-4c48-bb6d-c5ef01eba827" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 53 58" src="https://github.com/user-attachments/assets/ea9d73bb-027e-4f3f-ae0d-b27fedaaf61d" />
<img width="1561" alt="Screenshot 2025-05-23 at 09 54 08" src="https://github.com/user-attachments/assets/f34858ca-14d8-4091-aa78-05ff8150defe" />
