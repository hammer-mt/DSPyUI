# DSPyUI - Gradio Spaces Deployment Guide

This guide explains how to deploy DSPyUI to Hugging Face Spaces.

## Quick Deploy

### Option 1: Deploy from GitHub

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose:
   - **Space name**: dspyui (or your preferred name)
   - **License**: Apache 2.0
   - **Space SDK**: Gradio
   - **Space hardware**: CPU Basic (free tier works!)
4. Link your GitHub repository or upload files:
   - `app.py`
   - `interface.py`
   - `core.py`
   - `models.py`
   - `requirements.txt`
   - `example_data/` directory
5. Spaces will automatically detect `app.py` and launch the application

### Option 2: Deploy via Hugging Face CLI

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Login to Hugging Face
huggingface-cli login

# Create a new Space
huggingface-cli repo create --type space --space-sdk gradio dspyui

# Clone the Space repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/dspyui
cd dspyui

# Copy DSPyUI files
cp /path/to/DSPyUI/{app.py,interface.py,core.py,models.py,requirements.txt} .
cp -r /path/to/DSPyUI/example_data .

# Commit and push
git add .
git commit -m "Initial DSPyUI deployment"
git push
```

## Using DSPyUI on Spaces

### Setting Up API Keys

**DSPyUI on Spaces does NOT require a `.env` file!** Instead:

1. Navigate to the **Settings** tab in the application
2. Enter your API keys for the providers you want to use:
   - **OpenAI** (for GPT models)
   - **Anthropic** (for Claude models)
   - **Groq** (for Groq models)
   - **Google** (for Gemini models)
3. Click **Save API Keys**
4. API keys are stored in your browser session only and never saved to disk

### Security Notes

✅ **Your API keys are secure:**
- Keys are stored in session state only
- Keys are never saved to disk or logs
- Keys are cleared when you close your browser
- Each user session is isolated

⚠️ **Important:**
- You'll need to re-enter API keys in each new session
- Don't share your Space URL with API keys already configured
- Consider using Spaces Secrets for permanent deployment (see below)

### Using Local LLMs (No API Keys Required!)

If you don't have API keys, you can still use DSPyUI with local LLM servers:

1. Set up LM Studio, Ollama, or llama.cpp locally
2. Note your server URL (e.g., `http://localhost:1234/v1`)
3. In DSPyUI, select a model starting with `local:`
4. Enter your server URL when prompted
5. Compile programs using your local LLM

**Note:** Local LLMs won't work on Spaces (server must be accessible from Spaces). This is only for local deployment.

## Advanced: Using Spaces Secrets (Recommended for Persistent Deployment)

For a more permanent deployment where API keys don't need to be re-entered:

1. Go to your Space settings on Hugging Face
2. Navigate to **Repository secrets**
3. Add secrets:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GROQ_API_KEY`
   - `GOOGLE_API_KEY`
4. Modify `app.py` to load secrets:

```python
import os
from huggingface_hub import HfFolder

# Load secrets from Spaces environment
for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY"]:
    secret_value = os.environ.get(key)
    if secret_value:
        os.environ[key] = secret_value
```

5. Users can still override these with the Settings tab

## Hardware Requirements

### CPU (Free Tier)

- ✅ Works fine for most use cases
- ⚠️ Compilation may be slow with large datasets
- Suitable for: up to 100 training examples

### CPU Upgrade ($0.05/hr)

- ✅ Faster compilation
- ✅ Handles larger datasets (1000+ examples)
- Recommended for: heavy usage, MIPRO optimizer

### GPU (T4 - $0.60/hr)

- Not required for DSPyUI (LLM inference happens via API)
- Only needed if running local LLMs on Spaces (not recommended)

## File Structure for Spaces

```
your-space/
├── app.py                  # Spaces entry point
├── interface.py            # Gradio UI
├── core.py                 # DSPy logic
├── models.py              # Pydantic models
├── requirements.txt        # Python dependencies
├── example_data/          # Demo datasets
│   ├── jokes.csv
│   ├── jokes2.csv
│   └── jokes3.csv
└── README.md              # This file (optional, for Space description)
```

**Not needed on Spaces:**
- `.env` (use Settings tab or Spaces secrets)
- `webui.sh` (use `app.py` instead)
- `dspy/` (installed from PyPI)
- `.agent/`, `tests/`, etc. (development files)

## Troubleshooting

### "No module named 'gradio'"

Ensure `requirements.txt` includes all dependencies. Should contain:

```
gradio>=5.0.0
dspy-ai>=3.0.0
pandas>=2.0.0
openai>=1.0.0
anthropic>=0.40.0
groq>=0.12.0
google-generativeai>=0.8.0
scikit-learn>=1.3.0
pydantic>=2.0.0
```

### "API key not found"

1. Check you've entered keys in the Settings tab
2. Click "Save API Keys" button
3. Verify the status shows keys as "Configured"
4. If using Spaces secrets, check they're set in repository settings

### Space keeps restarting

- Check Spaces logs for errors
- Ensure all imports are available in `requirements.txt`
- Verify Python version compatibility (3.10+)

### Compilation fails with timeout

- Use smaller datasets on free tier
- Reduce `max_iters` parameter
- Use simpler optimizers (BootstrapFewShot instead of MIPROv2)
- Consider upgrading to CPU Upgrade tier

## Example Space Configuration

**README.md** (for your Space):

```markdown
---
title: DSPyUI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.48.0
app_file: app.py
pinned: false
license: apache-2.0
---

# DSPyUI

A visual interface for DSPy - making DSPy accessible to everyone!

## How to Use

1. Go to **Settings** tab
2. Enter your API keys (OpenAI, Anthropic, etc.)
3. Click Save
4. Go to **Compile Program** tab
5. Define your task and upload example data
6. Click Compile!

No API keys? No problem! You can explore the demo programs in the **View Prompts** tab.
```

## Support

- GitHub Issues: [DSPyUI Issues](https://github.com/hammer-mt/DSPyUI/issues)
- DSPy Documentation: [DSPy Docs](https://dspy-docs.vercel.app)
- Hugging Face Spaces Docs: [Spaces Documentation](https://huggingface.co/docs/hub/spaces)

## License

Apache 2.0 - See LICENSE file
