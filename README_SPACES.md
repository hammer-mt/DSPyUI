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
   - `requirements.txt`
   - `dspyui/` directory (the app itself, including `dspyui/example_data/`)
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
cp /path/to/DSPyUI/{app.py,requirements.txt} .
cp -r /path/to/DSPyUI/dspyui .

# Commit and push
git add .
git commit -m "Initial DSPyUI deployment"
git push
```

## Using DSPyUI on Spaces

### Setting Up API Keys

**DSPyUI on Spaces does NOT require a `.env` file!** Instead:

1. Navigate to the **Settings** tab in the application
2. Enter your API key. **OpenRouter is recommended**: one key works for every model in the Model and Teacher lists. The others are optional, for calling a provider directly:
   - **OpenRouter** (recommended)
   - **OpenAI** (for GPT models)
   - **Anthropic** (for Claude models)
   - **Groq** (for Groq models)
   - **Google** (for Gemini models)
3. Click **Save API Keys**
4. API keys are stored in your browser session only and never saved to disk

### Security Notes

✅ **Your API keys are secure:**
- Keys are stored in session state only (on the server, for your session)
- Keys are never saved to disk or logs, and never written to the server's environment
- Keys are cleared when you close your browser
- Each user session is isolated: other visitors cannot see or use the keys you enter
- Keys that the Space owner configured as secrets are never sent to anyone's browser; the Settings tab only shows whether a provider is configured

⚠️ **Important:**
- You'll need to re-enter API keys in each new session
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
3. Add secrets. `OPENROUTER_API_KEY` on its own is enough (recommended); the rest are optional:
   - `OPENROUTER_API_KEY`
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

No code changes are needed: Spaces passes secrets to the app as environment variables, which is where DSPyUI looks for keys.

4. Users can still use their own key for a provider by entering it in the Settings tab (it applies to their session only)

⚠️ **Secrets are shared with everyone who can open the Space.** Visitors cannot read the keys, but every program a visitor compiles or runs is billed to the Space owner's keys. Keep a Space with secrets **private**, or leave the secrets out and let each visitor enter their own keys.

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
├── requirements.txt        # Python dependencies
├── dspyui/
│   ├── interface.py        # Gradio UI
│   ├── core.py             # DSPy logic
│   └── example_data/       # Demo datasets
│       ├── rating_jokes.csv
│       ├── telling_jokes.csv
│       └── rewriting_jokes.csv
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
gradio>=5.50.0,<6
dspy>=3.3.0
optuna>=3.4.0
pandas>=2.0.0
numpy>=1.26
python-dotenv>=1.0.0
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
sdk_version: 5.50.0
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
