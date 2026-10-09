"""
DSPyUI core: everything that talks to DSPy.

READING GUIDE (for readers of "Context Engineering with DSPy")
==============================================================
Chapter 10, "Building a Gradio User Interface (DSPyUI)", and the companion
notebook `chapter10/dspyui-gradio.ipynb` walk through a compact version of this
app. This file is the full version of the same building blocks, in the same
order as the chapter. Search for "BOOK:" to jump between the sections:

    Book section     What to look for here
    ---------------  ---------------------------------------------------------
    Signatures       create_custom_signature()
    Modules          MODEL_OPTIONS, create_module()
    Datasets         split_dataframe(), load_and_split(), load_manual_and_split()
    Optimization     create_metric(), compile_program()   (Table 10-1 optimizers)
    Inference        run_inference()

The functions above have the same names and arguments as the ones printed in
the book. Where the full app needs more than the book's simplified version
(extra modules, extra optimizers, saving, cost tracking), the additions are
marked "Beyond the book".

Everything below "APP PLUMBING" is glue the Gradio interface (interface.py)
needs: turning the web form into calls to the functions above, saving
artifacts to disk, listing saved prompts, cost tracking, and chains.
"""

import contextlib
import contextvars
import datetime
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import dspy
import numpy as np
import pandas as pd
from dspy.signatures.signature import make_signature

from dspyui import __version__ as DSPYUI_VERSION

# Example data ships inside the package so the demo buttons work after
# `pip install dspyui-gradio` as well as from a git checkout.
PACKAGE_DIR = Path(__file__).resolve().parent
EXAMPLE_DATA_DIR = PACKAGE_DIR / "example_data"

# The pre-compiled joke judge that ships with DSPyUI. It is used as the default
# "Judge Prompt" for the LLM-as-a-Judge metric (see the "Telling Jokes" demo).
EXAMPLE_JUDGE_ID = "JokeTopic:Funny-Gpt4oMini_ChainOfThought_Bootstrapfewshotwithrandomsearch-20241003"


# =============================================================================
# BOOK: Signatures
# =============================================================================
# The Signature builder is the heart of DSPyUI. The web form collects input
# field names/descriptions, output field names/descriptions and the task
# instructions; this function turns them into a real dspy.Signature.
#
# `make_signature` is DSPy's programmatic way to create Signatures at runtime,
# as opposed to the class-based syntax (`class QA(dspy.Signature): ...`) you
# use when writing code by hand. This is the function printed in the book.

def create_custom_signature(
    input_fields,
    output_fields,
    instructions,
    input_descs,
    output_descs,
):
    fields = {}

    for i, name in enumerate(input_fields):
        desc = (
            input_descs[i]
            if i < len(input_descs) and input_descs[i]
            else None
        )
        fields[name] = (
            str,
            # The book prints `dspy.InputField(desc=desc)`. The description is
            # only passed when there is one, because desc=None would put the
            # word "None" into the prompt as the field's description.
            dspy.InputField(desc=desc) if desc else dspy.InputField(),
        )

    for i, name in enumerate(output_fields):
        desc = (
            output_descs[i]
            if i < len(output_descs) and output_descs[i]
            else None
        )
        fields[name] = (
            str,
            dspy.OutputField(desc=desc) if desc else dspy.OutputField(),
        )

    return make_signature(
        fields,
        instructions=instructions,
    )


# =============================================================================
# BOOK: Modules
# =============================================================================
# The user picks a student model (runs in production, so it should be cheap)
# and a teacher model (generates demonstrations / reflects during optimization,
# so it should be good). Names use DSPy's "provider/model" format and are
# passed straight to dspy.LM(...).
#
# The first four entries are the list printed in the book. The entries after
# them are the "..." -- more current models, the same models through
# OpenRouter, and local OpenAI-compatible servers.
#
# API keys: the recommended setup is a single OPENROUTER_API_KEY. It works for
# every hosted model below, including the "openai/", "anthropic/", "gemini/"
# and "groq/" ones: when a provider's own key is not set, make_lm() sends that
# model through OpenRouter. Set a provider's own key only if you want to call
# it directly.

MODEL_OPTIONS = [
    "openai/gpt-5.6-luna",
    "openai/gpt-5.6-sol",
    "anthropic/claude-opus-4.8",
    "gemini/gemini-3.5-flash",
    # --- Beyond the book: more models ---------------------------------------
    "openai/gpt-5.6-terra",
    "anthropic/claude-sonnet-5.5",
    "anthropic/claude-haiku-5.5",
    "gemini/gemini-3.8-flash",
    "groq/openai/gpt-oss-120b",
    # OpenRouter (recommended): needs only OPENROUTER_API_KEY
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/openai/gpt-5.6-sol",
    "openrouter/anthropic/claude-opus-4.8",
    "openrouter/google/gemini-3.5-flash",
    "openrouter/moonshotai/kimi-k2.6",
    "openrouter/meta-llama/llama-3.1-8b-instruct",
    # Local LLMs (LM Studio, Ollama, llama.cpp): "local:<model name>"
    "local:qwen",
    "local:llama",
    "local:custom",
]

# "In my projects, I default to gpt-5.6-sol as teacher and gpt-5.6-luna as student."
DEFAULT_STUDENT_MODEL = "openai/gpt-5.6-luna"
DEFAULT_TEACHER_MODEL = "openai/gpt-5.6-sol"

MODULE_OPTIONS = ["Predict", "ChainOfThought", "ChainOfThoughtWithHint", "ProgramOfThought"]


def create_module(module_name, signature, hint=None, max_iters=3):
    # This is the factory function from the book. dspy.Predict and
    # dspy.ChainOfThought are already dspy.Module subclasses, so they work
    # directly with compile(), save(), and load(). No wrapper classes needed.
    if module_name == "Predict":
        return dspy.Predict(signature)
    elif module_name == "ChainOfThought":
        return dspy.ChainOfThought(signature)

    # --- Beyond the book: two extra modules in the Module dropdown ----------
    elif module_name == "ChainOfThoughtWithHint":
        return ChainOfThoughtWithHint(signature, hint)
    elif module_name == "ProgramOfThought":
        # Writes and runs Python to produce the answer (requires deno).
        return dspy.ProgramOfThought(signature, max_iters=max_iters)

    raise ValueError(f"Unsupported DSPy module: {module_name}")


class ChainOfThoughtWithHint(dspy.Module):
    """ChainOfThought with a fixed, user-supplied hint.

    DSPy removed its built-in ChainOfThoughtWithHint, so DSPyUI rebuilds it the
    way you would by hand: add a `hint` input field to the Signature and pass
    the same hint on every call.
    """

    def __init__(self, signature, hint=None):
        super().__init__()
        self.hint = hint or ""
        self.cot = dspy.ChainOfThought(
            signature.prepend(
                "hint",
                dspy.InputField(desc="A hint to guide the reasoning"),
                type_=str,
            )
        )

    def forward(self, **kwargs):
        kwargs["hint"] = self.hint
        return self.cot(**kwargs)


# Model names saved by older versions of DSPyUI had no provider prefix.
_LEGACY_GROQ_MODELS = ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]

# Which environment variable holds the key for each provider prefix.
PROVIDER_KEY_ENV = {
    "openrouter": ["OPENROUTER_API_KEY"],  # recommended: one key for every provider
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],  # the book uses GEMINI_API_KEY
    "groq": ["GROQ_API_KEY"],
}


def key_is_set(value: Optional[str]) -> bool:
    """True for a real key; False for empty values and the .env.example placeholder."""
    return bool(value and value.strip() and value.strip() != "fillthisin")


# API keys typed into the Settings tab belong to one browser session. They are
# kept out of os.environ (which every visitor of a shared deployment would see
# the effect of) and handed to DSPy only for that session's own requests.
_session_api_keys: contextvars.ContextVar[Dict[str, str]] = contextvars.ContextVar(
    "dspyui_session_api_keys", default={}
)


@contextlib.contextmanager
def use_api_keys(api_keys: Optional[Dict[str, str]]):
    """Use these API keys (e.g. {"OPENAI_API_KEY": "sk-..."}) for the LMs created inside the block.

    They take priority over keys in the environment and are forgotten when the
    block ends. The interface wraps each button click in this.
    """
    token = _session_api_keys.set({name: key for name, key in (api_keys or {}).items() if key_is_set(key)})
    try:
        yield
    finally:
        _session_api_keys.reset(token)


def _session_key(provider: str) -> Optional[str]:
    """This session's key for a provider (from the Settings tab), if it entered one."""
    session_keys = _session_api_keys.get()
    for env_name in PROVIDER_KEY_ENV.get(provider, []):
        if session_keys.get(env_name):
            return session_keys[env_name]
    return None


def _provider_key(provider: str) -> Optional[str]:
    """The key to use for a provider: this session's own key, else the environment's."""
    session_key = _session_key(provider)
    if session_key:
        return session_key
    for env_name in PROVIDER_KEY_ENV.get(provider, []):
        if key_is_set(os.environ.get(env_name)):
            return os.environ[env_name]
    return None


def _explicit_key(provider: str) -> Dict[str, str]:
    """`api_key=...` for dspy.LM when DSPy would not find the right key by itself.

    Keys in the environment are picked up by DSPy without help. A key from the
    Settings tab has to be passed in, and so does GOOGLE_API_KEY (the older
    name DSPyUI used for what DSPy calls GEMINI_API_KEY).
    """
    session_key = _session_key(provider)
    if session_key:
        return {"api_key": session_key}
    if provider == "gemini" and not key_is_set(os.environ.get("GEMINI_API_KEY")) and key_is_set(os.environ.get("GOOGLE_API_KEY")):
        return {"api_key": os.environ["GOOGLE_API_KEY"]}
    return {}


def normalize_model_name(model: str) -> str:
    """Turn any model name the UI accepts into DSPy's "provider/model" format."""
    model = (model or "").strip()
    if not model or model.startswith("local:"):
        return model

    if "/" not in model:
        # Bare names from prompts saved by older versions of DSPyUI.
        if model.startswith("claude"):
            model = f"anthropic/{model}"
        elif model.startswith("gemini"):
            model = f"gemini/{model}"
        elif model in _LEGACY_GROQ_MODELS:
            model = f"groq/{model}"
        else:
            model = f"openai/{model}"

    if model.startswith("google/"):
        # LiteLLM calls the Gemini API provider "gemini", not "google".
        model = "gemini/" + model[len("google/"):]

    if model.startswith("anthropic/"):
        # The book writes "anthropic/claude-opus-4.8"; Anthropic's API spells
        # the version with a hyphen ("claude-opus-4-8"). Accept either.
        model = re.sub(r"(\d)\.(\d)", r"\1-\2", model)

    return model


def _openrouter_equivalent(model: str) -> Optional[str]:
    """The OpenRouter slug for a direct-provider model, if there is one."""
    provider, _, name = model.partition("/")
    if provider == "openai":
        return f"openrouter/openai/{name}"
    if provider == "anthropic":
        # OpenRouter spells versions with a dot and no date suffix.
        name = re.sub(r"-\d{8}$", "", name)
        name = re.sub(r"(\d)-(\d)(?!\d)", r"\1.\2", name)
        return f"openrouter/anthropic/{name}"
    if provider == "gemini":
        return f"openrouter/google/{name}"
    if provider == "groq" and "/" in name:
        # e.g. groq/openai/gpt-oss-120b: the same open model, served via OpenRouter
        return f"openrouter/{name}"
    return None


def make_lm(model: str, base_url: Optional[str] = None, **kwargs) -> dspy.LM:
    """Create the dspy.LM for a model picked in the Model / Teacher dropdowns.

    In the book this is simply `dspy.LM("openai/gpt-5.6-luna")`. The extra
    work here supports local servers, older saved prompts, and the recommended
    setup of a single OpenRouter key: a model whose own provider key is not
    set is sent through OpenRouter instead.
    """
    if not model:
        raise ValueError("Please select a model.")

    if model.startswith("local:"):
        # Local LLM behind an OpenAI-compatible endpoint (LM Studio, Ollama, llama.cpp).
        return dspy.LM(
            f"openai/{model[len('local:'):]}",
            api_base=base_url or "http://127.0.0.1:1234/v1",
            api_key=_provider_key("openai") or "local",
            **kwargs,
        )

    model = normalize_model_name(model)
    provider = model.split("/", 1)[0]

    if provider in PROVIDER_KEY_ENV and provider != "openrouter" and not _provider_key(provider):
        # No key for this provider, so use the same model through OpenRouter
        # (the recommended one-key setup).
        fallback = _openrouter_equivalent(model)
        if fallback and _provider_key("openrouter"):
            print(f"{PROVIDER_KEY_ENV[provider][0]} is not set - running {model} through OpenRouter as {fallback}")
            return dspy.LM(fallback, **{**_explicit_key("openrouter"), **kwargs})

    return dspy.LM(model, **{**_explicit_key(provider), **kwargs})


def _as_lm(lm_or_model: Union[dspy.LM, str, None]) -> Optional[dspy.LM]:
    return make_lm(lm_or_model) if isinstance(lm_or_model, str) else lm_or_model


# A default LM, so that using these functions from a script or notebook works
# without extra setup. The web app never relies on it: Gradio runs every button
# click in a worker thread, and DSPy only lets the thread that first called
# dspy.configure() call it again. So everywhere below uses
# `with dspy.context(lm=...)` instead, which is thread-safe.
if dspy.settings.lm is None:
    try:
        dspy.configure(lm=dspy.LM(DEFAULT_STUDENT_MODEL))
    except Exception as e:  # pragma: no cover - only if DSPy is already configured elsewhere
        print(f"Note: could not set a default DSPy LM ({e})")


# =============================================================================
# BOOK: Datasets
# =============================================================================
# Users upload a CSV file with columns matching their Signature fields (or type
# rows into the table by hand). The data is mapped to dspy.Example objects and
# split 80/20 into train and test sets.

def split_dataframe(df, input_fields, output_fields):
    fields = input_fields + output_fields
    missing = [field for field in fields if field not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    # Skip blank rows (the "Enter manually" table starts with an empty one)
    blank = df[fields].apply(lambda column: column.map(lambda v: pd.isna(v) or str(v).strip() == ""))
    df = df[~blank.all(axis=1)].fillna("")
    if len(df) < 2:
        raise ValueError("At least two examples are required for compilation.")

    dataset = [
        dspy.Example(
            **{f: str(row[f]) for f in input_fields + output_fields}
        ).with_inputs(*input_fields)
        for _, row in df.iterrows()
    ]

    # The book shows `split = int(0.8 * len(dataset))`. The min/max only matter
    # for tiny datasets: they guarantee neither side of the split is empty.
    split = min(max(1, int(0.8 * len(dataset))), len(dataset) - 1)
    return dataset[:split], dataset[split:]


def load_and_split(file_path, input_fields, output_fields):
    # In the book `file_path` is always a CSV path. The app also passes the
    # table from the screen (already a DataFrame), so accept both.
    df = file_path if isinstance(file_path, pd.DataFrame) else pd.read_csv(file_path)
    return split_dataframe(df, input_fields, output_fields)


def load_manual_and_split(rows, input_fields, output_fields):
    # "Enter manually" mode: rows typed straight into the gr.Dataframe table.
    fields = input_fields + output_fields
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df = df.dropna(how="all")
    if list(df.columns) != fields:
        if len(df.columns) != len(fields):
            raise ValueError(
                f"Manual data needs {len(fields)} columns in this order: "
                + ", ".join(fields)
            )
        df.columns = fields
    return split_dataframe(df, input_fields, output_fields)


# =============================================================================
# BOOK: Optimization -- metrics
# =============================================================================
# The metric options are exact match, cosine similarity, and LLM-as-a-judge
# (Chapter 5). Every metric accepts the five arguments GEPA needs
# (gold, pred, trace, pred_name, pred_trace), so the same function works with
# all of the optimizers in Table 10-1.

METRIC_OPTIONS = ["Exact Match", "Cosine Similarity", "LLM-as-a-Judge"]


def _field(obj: Any, field: str, default: Any = "") -> Any:
    """Read a field from a dspy.Example, a dspy.Prediction, or a plain dict."""
    if isinstance(obj, dict):
        return obj.get(field, default)
    try:
        return obj[field]
    except Exception:
        return getattr(obj, field, default)


def _has_field(obj: Any, field: str) -> bool:
    try:
        return obj is not None and field in obj
    except TypeError:
        return hasattr(obj, field)


def _make_embedder() -> dspy.Embedder:
    # Cosine Similarity uses OpenAI embeddings. Without an OpenAI key, fall back
    # to the same embedding model through OpenRouter.
    if not _provider_key("openai") and _provider_key("openrouter"):
        return dspy.Embedder("openrouter/openai/text-embedding-3-small", **_explicit_key("openrouter"))
    return dspy.Embedder("openai/text-embedding-3-small", **_explicit_key("openai"))


def _parse_score(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if match is None:
            raise ValueError(f"Judge did not return a numeric score: {value!r}")
        return float(match.group())


def _judge_paths(judge_prompt_id: str) -> Tuple[Path, Callable[[str], Path]]:
    """Where a judge's prompt details and compiled program live on disk."""
    # The Judge Prompt dropdown shows file names, so drop a trailing ".json".
    judge_id = judge_prompt_id[:-5] if judge_prompt_id.endswith(".json") else judge_prompt_id
    if judge_id == EXAMPLE_JUDGE_ID:
        # The bundled example (stored without the ":" so it also works on Windows).
        safe_id = judge_id.replace(":", "-")
        return (
            EXAMPLE_DATA_DIR / f"{safe_id}.json",
            lambda _program_id: EXAMPLE_DATA_DIR / f"{safe_id}-program.json",
        )
    return Path("prompts") / f"{judge_id}.json", lambda program_id: Path("programs") / f"{program_id}.json"


def load_judge_program(judge_prompt_id: str) -> Tuple[dspy.Module, List[str], List[str]]:
    """Load a previously compiled DSPyUI program to act as the judge.

    Returns (judge_program, judge_input_fields, judge_output_fields).
    """
    details_path, program_path_for = _judge_paths(judge_prompt_id)
    if not details_path.exists():
        raise ValueError(f"Judge prompt not found: {details_path}")

    with open(details_path, "r") as f:
        details = json.load(f)

    judge_input_fields = details.get("input_fields", [])
    judge_output_fields = details.get("output_fields", [])
    judge_signature = create_custom_signature(
        judge_input_fields,
        judge_output_fields,
        details.get("instructions", ""),
        details.get("input_descs", details.get("input_descriptions", [])),
        details.get("output_descs", details.get("output_descriptions", [])),
    )
    judge_program = create_module(
        details.get("dspy_module", "Predict"),
        judge_signature,
        hint=details.get("hint"),
        max_iters=details.get("max_iters", 3),
    )

    program_path = program_path_for(details.get("human_readable_id", ""))
    if not program_path.exists():
        raise ValueError(f"Compiled judge program not found: {program_path}")
    load_program(judge_program, program_path)

    return judge_program, judge_input_fields, judge_output_fields


class JudgeOutputs(dspy.Signature):
    """Score how well a candidate matches the reference output."""

    reference: str = dspy.InputField()
    candidate: str = dspy.InputField()
    score: float = dspy.OutputField(desc="0.0 to 1.0")
    feedback: str = dspy.OutputField(desc="Specific reason for the score")


def create_metric(metric_name, output_fields, judge_model=None, judge_prompt_id=None):
    """Build the metric function for one of METRIC_OPTIONS.

    For LLM-as-a-Judge the app passes `judge_prompt_id`: the judge is a program
    you compiled earlier in DSPyUI (the "Judge Prompt" dropdown). If you call
    this yourself with only `judge_model`, you get the generic reference-vs-
    candidate judge from the book's notebook instead.
    """
    if metric_name == "Exact Match":
        def exact_match(gold, pred, trace=None, pred_name=None, pred_trace=None):
            if pred is None:
                return 0.0
            return float(all(
                str(_field(pred, field)).strip() == str(_field(gold, field)).strip()
                for field in output_fields
            ))
        return exact_match

    if metric_name == "Cosine Similarity":
        embedder = _make_embedder()

        def cosine_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            try:
                gold_texts = [str(_field(gold, field)) for field in output_fields]
                pred_texts = [str(_field(pred, field)) for field in output_fields]
                vectors = embedder(gold_texts + pred_texts)
                midpoint = len(output_fields)
                similarities = []
                for gold_vector, pred_vector in zip(vectors[:midpoint], vectors[midpoint:]):
                    denominator = np.linalg.norm(gold_vector) * np.linalg.norm(pred_vector)
                    similarities.append(
                        float(np.dot(gold_vector, pred_vector) / denominator)
                        if denominator else 0.0
                    )
                return float(np.mean(similarities))
            except Exception as e:
                print(f"Error in cosine similarity metric: {e}")
                return 0.0
        return cosine_metric

    if metric_name == "LLM-as-a-Judge":
        if judge_prompt_id:
            judge_program, judge_input_fields, judge_output_fields = load_judge_program(judge_prompt_id)
            if len(judge_output_fields) != 1:
                raise ValueError(
                    f"The judge must have exactly one (numeric score) output field, got: {judge_output_fields}"
                )
            score_field = judge_output_fields[0]

            def judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
                try:
                    # The judge looks at what the program just produced (e.g.
                    # the joke it wrote); any other judge input (e.g. the
                    # topic) comes from the example.
                    judge_input = {}
                    for field in judge_input_fields:
                        if field in output_fields and _has_field(pred, field):
                            judge_input[field] = _field(pred, field)
                        elif _has_field(gold, field):
                            judge_input[field] = _field(gold, field)
                        elif _has_field(pred, field):
                            judge_input[field] = _field(pred, field)
                        else:
                            print(f"Warning: judge input field '{field}' not found in gold or pred")
                            judge_input[field] = ""
                    # Judge demonstrations must not leak into optimizer traces.
                    with dspy.context(trace=[]):
                        result = judge_program(**judge_input)
                    return _parse_score(_field(result, score_field, None))
                except Exception as e:
                    print(f"Error in LLM-as-a-Judge metric: {e}")
                    return 0.0
            return judge_metric

        if judge_model is None:
            raise ValueError("Judge prompt ID is required for LLM-as-a-Judge metric")

        # The generic judge from the book's notebook.
        judge = dspy.ChainOfThought(JudgeOutputs)
        judge.set_lm(_as_lm(judge_model))

        def generic_judge_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            reference = {field: _field(gold, field) for field in output_fields}
            candidate = {field: _field(pred, field) for field in output_fields}
            with dspy.context(trace=[]):
                judgment = judge(
                    reference=json.dumps(reference, sort_keys=True),
                    candidate=json.dumps(candidate, sort_keys=True),
                )
            score = min(1.0, max(0.0, float(judgment.score)))
            return dspy.Prediction(score=score, feedback=judgment.feedback)
        return generic_judge_metric

    raise ValueError(f"Unknown metric type: {metric_name}")


def with_feedback(metric, output_fields):
    """Wrap a numeric metric so GEPA also gets text to reflect on.

    GEPA calls the metric with five arguments. A plain number works, but GEPA
    improves instructions by *reading* what went wrong, so we return
    dspy.Prediction(score=..., feedback=...) as the book recommends.
    """
    def feedback_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        result = metric(gold, pred, trace, pred_name, pred_trace)
        if isinstance(result, dspy.Prediction):
            return result  # already has feedback
        score = float(result)
        expected = {field: str(_field(gold, field)) for field in output_fields if _has_field(gold, field)}
        produced = {field: str(_field(pred, field)) for field in output_fields}
        feedback = f"Score: {score:.2f}. The program produced {json.dumps(produced)}."
        if expected:
            feedback += f" The reference output was {json.dumps(expected)}."
        return dspy.Prediction(score=score, feedback=feedback)
    return feedback_metric


# =============================================================================
# BOOK: Optimization -- compile
# =============================================================================
# Table 10-1 (DSPyUI optimizers), plus two extras the app has always offered:
#
#   Optimizer         Best for
#   ----------------  ------------------------------------------------
#   BootstrapFewShot  Small datasets (~10 examples), quick iteration
#   BootstrapRS       Medium datasets (50+), need reliable improvement
#   MIPROv2           Large datasets (100+), production optimization
#   GEPA              Creative/subjective tasks, instruction tuning
#   COPRO             (beyond the book) instruction-only search
#   LabeledFewShot    (beyond the book) k labeled examples, no optimization
#
# "BootstrapRS" is DSPy's short name for BootstrapFewShotWithRandomSearch.

OPTIMIZER_OPTIONS = ["BootstrapFewShot", "BootstrapRS", "MIPROv2", "GEPA", "COPRO", "LabeledFewShot"]

# Prompts saved by older versions of DSPyUI use the long name.
_OPTIMIZER_ALIASES = {"BootstrapFewShotWithRandomSearch": "BootstrapRS"}


def normalize_optimizer_name(optimizer_name: str) -> str:
    return _OPTIMIZER_ALIASES.get(optimizer_name, optimizer_name)


def compile_program(module, trainset, devset, optimizer_name, metric,
    teacher_lm, save_path="programs/latest.json", k=16):
    """The core compile function from the book.

    1. A baseline evaluation runs on the unoptimized module.
    2. The optimizer compiles the module against the training set.
    3. A final evaluation runs on the optimized module.
    4. The compiled program is saved, and both scores are returned.

    `teacher_lm` may be a dspy.LM or a model name such as "openai/gpt-5.6-sol".
    The student LM is whichever LM is active (dspy.configure / dspy.context).
    """
    optimizer_name = normalize_optimizer_name(optimizer_name)
    teacher_lm = _as_lm(teacher_lm)

    # Baseline
    baseline_eval = dspy.Evaluate(metric=metric, devset=devset, num_threads=1)
    baseline_result = baseline_eval(module)
    baseline_score = baseline_result.score
    teacher = module.deepcopy()
    teacher.set_lm(teacher_lm)

    # Optimize
    if optimizer_name == "BootstrapFewShot":
        optimizer = dspy.BootstrapFewShot(metric=metric)
        compiled = optimizer.compile(
            module,
            teacher=teacher,
            trainset=trainset,
        )
    elif optimizer_name == "BootstrapRS":
        optimizer = dspy.BootstrapFewShotWithRandomSearch(metric=metric)
        compiled = optimizer.compile(
            module,
            teacher=teacher,
            trainset=trainset,
            valset=devset,
        )
    elif optimizer_name == "MIPROv2":
        # prompt_model writes the candidate instructions, so use the teacher.
        optimizer = dspy.MIPROv2(metric=metric, auto="light", prompt_model=teacher_lm)
        compiled = optimizer.compile(
            module,
            teacher=teacher,
            trainset=trainset,
            valset=devset,
        )
    elif optimizer_name == "GEPA":
        output_fields = [
            name for name in module.named_predictors()[-1][1].signature.output_fields
            if name != "reasoning"
        ]
        optimizer = dspy.GEPA(
            metric=with_feedback(metric, output_fields),
            auto="light",
            reflection_lm=teacher_lm,
        )
        compiled = optimizer.compile(
            module,
            trainset=trainset,
            valset=devset,
        )

    # ... other optimizers (beyond the book)
    elif optimizer_name == "COPRO":
        optimizer = dspy.COPRO(prompt_model=teacher_lm, metric=metric)
        compiled = optimizer.compile(
            module,
            trainset=trainset,
            eval_kwargs=dict(display_progress=True, display_table=0),
        )
    elif optimizer_name == "LabeledFewShot":
        optimizer = dspy.LabeledFewShot(k=k)
        compiled = optimizer.compile(module, trainset=trainset)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Final evaluation
    final_eval = dspy.Evaluate(metric=metric, devset=devset, num_threads=1)
    final_result = final_eval(compiled)
    final_score = final_result.score

    # DSPy's Evaluate returns a score as a percentage from 0 to 100.
    print(f"Baseline: {baseline_score:.2f}%, Optimized: {final_score:.2f}%")

    # Save the compiled program
    program_path = Path(save_path)
    program_path.parent.mkdir(parents=True, exist_ok=True)
    compiled.save(program_path)

    return baseline_score, final_score, compiled


# =============================================================================
# BOOK: Inference
# =============================================================================
# Load a saved program and run it. Returning the messages as well as the result
# is deliberate: users can see exactly what instructions and examples the
# optimizer produced, which builds trust in the system.

def run_inference(program_path, signature, module_name, input_data, hint=None, max_iters=3):
    # Re-create the module and load saved state
    module = create_module(module_name, signature, hint=hint, max_iters=max_iters)
    load_program(module, program_path)  # module.load(program_path), see below

    # Run inference
    result = module(**input_data)

    # Also extract the prompt for transparency
    messages = dspy.settings.lm.history[-1]["messages"]

    return result, messages


def load_program(module, program_path):
    """`module.load(program_path)`, plus support for old DSPyUI files.

    Programs saved by this version load with DSPy's own `module.load()`, exactly
    as in the book. Programs saved by DSPyUI before 2026 (DSPy 2.x, including
    the bundled example judge) used a different layout; for those we copy the
    few-shot demos and instructions across by hand.
    """
    program_path = Path(program_path)
    with open(program_path, "r") as f:
        state = json.load(f)

    predictors = module.named_predictors()
    native = all(
        isinstance(state.get(name), dict) and "signature" in state[name]
        for name, _ in predictors
    ) if not isinstance(module, dspy.Predict) else "signature" in state
    if native:
        module.load(program_path)
        return module

    # Legacy file: one saved state per predictor, in order.
    legacy_states = [v for v in state.values() if isinstance(v, dict) and "demos" in v]
    if "demos" in state:
        legacy_states = [state]
    for (_, predictor), legacy in zip(predictors, legacy_states):
        demos = []
        for demo in legacy.get("demos", []):
            demo = dict(demo)
            if "rationale" in demo:  # ChainOfThought's output was renamed in DSPy 2.5
                demo["reasoning"] = demo.pop("rationale")
            demos.append(dspy.Example(**demo))
        predictor.demos = demos
        if isinstance(legacy.get("signature"), dict):
            predictor.signature = predictor.signature.load_state(legacy["signature"])
        elif legacy.get("signature_instructions"):
            predictor.signature = predictor.signature.with_instructions(legacy["signature_instructions"])
    return module


def messages_to_text(messages: List[Dict[str, Any]]) -> str:
    """Flatten chat messages into the text shown in the 'Optimized Prompt' box."""
    text = ""
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, indent=2, default=str)
        text += f"{content}\n"
    return text


# #############################################################################
# APP PLUMBING
# Everything below turns the web form into calls to the functions above and
# keeps track of files on disk. It is not covered in the book.
# #############################################################################

# The student LM of each program compiled against a local model, so that
# "Generate" keeps using the same server (its URL is not saved with the program).
# Only local models are remembered: other LMs may carry a visitor's own API key.
_SESSION_LMS: Dict[str, dspy.LM] = {}


def generate_human_readable_id(input_fields: List[str], output_fields: List[str], dspy_module: str, llm_model: str, teacher_model: str, optimizer: str, instructions: str) -> str:
    # Create a signature-based name
    signature = '_'.join(input_fields + [':'] + output_fields)
    signature_pascal = ''.join(word.capitalize() for word in signature.split('_'))

    # "openai/gpt-5.6-luna" -> "Gpt56Luna" (drop the provider, keep it filename-safe)
    model_words = re.split(r'[^A-Za-z0-9]+', llm_model.split('/')[-1])
    model_name = ''.join(word.capitalize() for word in model_words if word)
    module_name = dspy_module
    optimizer_name = ''.join(word.capitalize() for word in optimizer.replace('bootstrap', 'bs').replace('randomsearch', 'rs').split('_'))

    # Get current date
    current_date = datetime.date.today().strftime("%Y%m%d")

    # Create a human-readable ID with date
    unique_id = f"{signature_pascal}-{model_name}_{module_name}_{optimizer_name}-{current_date}"

    if os.name == "nt":
        # ":" is not allowed in Windows file names
        unique_id = unique_id.replace(":", "-")

    return unique_id


def compile_from_ui(
    input_fields: List[str],
    output_fields: List[str],
    dspy_module: str,
    llm_model: str,
    teacher_model: str,
    example_data: pd.DataFrame,
    optimizer: str,
    instructions: str,
    metric_type: str,
    judge_prompt_id: Optional[str] = None,
    input_descs: Optional[List[str]] = None,
    output_descs: Optional[List[str]] = None,
    hint: Optional[str] = None,
    max_iters: int = 3,
    k: int = 16,
    llm_base_url: Optional[str] = None,
    teacher_base_url: Optional[str] = None
) -> Tuple[str, str, Dict[str, Any]]:
    """What the "Compile Program" button runs.

    This is the book's workflow end to end: Signature -> Module -> Dataset ->
    Metric -> compile_program(), followed by saving everything to disk.
    """
    # Models (BOOK: Modules)
    student_lm = make_lm(llm_model, llm_base_url)
    teacher_lm = make_lm(teacher_model, teacher_base_url or llm_base_url)

    # Signature and module (BOOK: Signatures, Modules)
    signature = create_custom_signature(input_fields, output_fields, instructions, input_descs or [], output_descs or [])
    module = create_module(dspy_module, signature, hint=hint, max_iters=max_iters)

    # Dataset, split 80/20 (BOOK: Datasets)
    trainset, devset = load_and_split(example_data, input_fields, output_fields)

    human_readable_id = generate_human_readable_id(input_fields, output_fields, dspy_module, llm_model, teacher_model, optimizer, instructions)
    program_path = f"programs/{human_readable_id}.json"

    # The student LM is active for everything inside this block. (The book calls
    # dspy.configure(lm=...); dspy.context is the thread-safe equivalent, which
    # a web app needs.)
    with dspy.context(lm=student_lm):
        # Metric, then baseline -> optimize -> final evaluation (BOOK: Optimization)
        metric = create_metric(metric_type, output_fields, judge_prompt_id=judge_prompt_id)
        baseline_score, score, compiled_program = compile_program(
            module,
            trainset,
            devset,
            optimizer,
            metric,
            teacher_lm,
            save_path=program_path,
            k=k,
        )
        print(f"Compiled program saved to {program_path}")

        # Use the compiled program with the first row of example data, to show
        # the full prompt that is sent to the LM (BOOK: Inference)
        first_row = example_data.iloc[0]
        input_data = {field: str(first_row[field]) for field in input_fields}
        result = compiled_program(**input_data)
        final_prompt = messages_to_text(dspy.settings.lm.history[-1]['messages'])

    if llm_model.startswith("local:"):
        _SESSION_LMS[human_readable_id] = student_lm

    # As in the book, programs/latest.json is always the most recent compile.
    try:
        shutil.copyfile(program_path, "programs/latest.json")
    except OSError as e:
        print(f"Warning: could not update programs/latest.json: {e}")

    # Calculate actual cost from DSPy history
    actual_cost_data = calculate_actual_cost(llm_model, teacher_model, lms=[student_lm, teacher_lm])
    print(f"Actual cost: ${actual_cost_data['actual_cost_usd']:.4f}")
    print(f"Total tokens: {actual_cost_data['total_tokens']:,} ({actual_cost_data['input_tokens']:,} input, {actual_cost_data['output_tokens']:,} output)")

    # Save the dataframe to the datasets folder
    os.makedirs('datasets', exist_ok=True)
    dataset_path = os.path.join('datasets', f"{human_readable_id}.csv")
    example_data.to_csv(dataset_path, index=False)
    print(f"Dataset saved to {dataset_path}")

    usage_instructions = f"""Program compiled successfully!
Evaluation score: {score}
Baseline score: {baseline_score}
The compiled program has been saved as 'programs/{human_readable_id}.json'.
You can now use the compiled program as follows:

compiled_program = dspy.{dspy_module}(CustomSignature)
compiled_program.load('programs/{human_readable_id}.json')
result = compiled_program({', '.join(f'{field}=value' for field in input_fields)})
print({', '.join(f'result.{field}' for field in output_fields)})
"""

    # Update the usage instructions to include the hint if applicable
    if dspy_module == "ChainOfThoughtWithHint":
        usage_instructions += f"\nHint: {hint}\n"

    example_output = f"\nExample usage with first row of data:\n"
    example_output += f"Input: {input_data}\n"
    example_output += f"Output: {result}\n"
    usage_instructions += example_output

    return usage_instructions, final_prompt, actual_cost_data


# Function to list prompts
def list_prompts(
    signature_filter: Optional[str] = None,
    output_filter: Optional[List[str]] = None
) -> List[Dict[str, Any]]:

    if not os.path.exists('prompts'):
        print("Prompts directory does not exist")
        return []

    files = os.listdir('prompts')
    if not files:
        print("No prompt files found in the prompts directory")
        return []

    prompt_details = []
    for file in files:
        if file.endswith('.json'):
            with open(os.path.join('prompts', file), 'r') as f:
                data = json.load(f)
                prompt_id = file
                signature = f"{', '.join(data['input_fields'])} -> {', '.join(data['output_fields'])}"

                input_signature = f"{', '.join(data['input_fields'])}"

                eval_score = data.get('evaluation_score', 'N/A')
                actual_cost = data.get('actual_cost') or 0
                total_tokens = data.get('total_tokens') or 0

                # Exclude example data
                details = {k: v for k, v in data.items() if k != 'example_data'}

                # Check if signature_filter is provided and matches
                if signature_filter and signature_filter.lower() not in signature.lower():
                    print(f"Skipping file {file} due to signature mismatch")
                    continue

                # Check if output_filter is provided and matches
                if output_filter:
                    if not all(filter_item.lower() in input_signature.lower() for filter_item in output_filter):
                        continue

                prompt_details.append({
                    "ID": prompt_id,
                    "Signature": signature,
                    "Eval Score": eval_score,
                    "Cost": actual_cost,
                    "Tokens": total_tokens,
                    "Details": json.dumps(details, indent=4)  # Add full details as a JSON string
                })

    print(f"Found {len(prompt_details)} saved prompts")
    return prompt_details  # Return the list of prompts as dictionaries

def load_example_csv(example_name: str) -> Optional[pd.DataFrame]:
    """Load an example CSV file from the bundled example_data directory."""
    csv_path = EXAMPLE_DATA_DIR / f"{example_name}.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return None


def export_to_csv(data: List[Dict[str, Any]]) -> str:
    """Export data to a CSV file."""
    df = pd.DataFrame(data)
    filename = "exported_data.csv"
    df.to_csv(filename, index=False)
    return filename


# -----------------------------------------------------------------------------
# Cost tracking
# -----------------------------------------------------------------------------

# Fallback pricing in USD per 1M tokens (input, output), used only when DSPy
# does not report a cost for a call. Current models are priced from LiteLLM's
# registry (see get_model_pricing); this table covers local models and the
# older models found in prompts saved by earlier versions of DSPyUI.
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # Anthropic models
    "claude-3-5-sonnet-20240620": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},

    # Local models have zero cost
    "local": {"input": 0.0, "output": 0.0}
}


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing information for a model.

    Args:
        model: Model name (e.g., 'openai/gpt-5.6-luna', 'openrouter/openai/gpt-5.6-sol')

    Returns:
        Dictionary with 'input' and 'output' prices per 1M tokens in USD
    """
    # Handle local models
    if model.startswith("local:"):
        return MODEL_PRICING["local"]

    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Current models: ask LiteLLM, which DSPy uses to call every provider
    try:
        import litellm

        normalized = normalize_model_name(model)
        for name in (normalized, normalized.split("/", 1)[-1]):
            info = litellm.model_cost.get(name)
            if info and info.get("input_cost_per_token") is not None:
                return {
                    "input": info["input_cost_per_token"] * 1_000_000,
                    "output": (info.get("output_cost_per_token") or 0.0) * 1_000_000,
                }
    except Exception as e:
        print(f"Warning: could not look up pricing for '{model}': {e}")

    # Unknown model: use the default student model's pricing as an estimate
    print(f"Warning: Unknown model '{model}', using {DEFAULT_STUDENT_MODEL} pricing as estimate")
    if model != DEFAULT_STUDENT_MODEL:
        return get_model_pricing(DEFAULT_STUDENT_MODEL)
    return {"input": 0.20, "output": 1.20}


def estimate_compilation_cost(
    dataset_size: int,
    optimizer: str,
    student_model: str,
    teacher_model: str,
    input_fields: List[str],
    output_fields: List[str],
    instructions: str
) -> Dict[str, Any]:
    """
    Estimate the cost of compiling a DSPy program.

    Args:
        dataset_size: Number of examples in the training dataset
        optimizer: Type of optimizer (e.g., 'BootstrapFewShot', 'MIPROv2')
        student_model: Model used for the student program
        teacher_model: Model used for teaching/optimization
        input_fields: List of input field names
        output_fields: List of output field names
        instructions: Task instructions

    Returns:
        Dictionary containing cost estimate details
    """
    optimizer = normalize_optimizer_name(optimizer)

    # Get pricing for both models
    student_pricing = get_model_pricing(student_model)
    teacher_pricing = get_model_pricing(teacher_model)

    # Estimate average token counts
    # Input: field names + instructions + example data (conservative estimate)
    avg_input_tokens = 100 + len(instructions.split()) * 1.3 + len(input_fields) * 20
    # Output: conservative estimate for generated response
    avg_output_tokens = 300

    # Estimate number of API calls based on optimizer type
    if optimizer == "LabeledFewShot":
        # No optimization calls, just inference
        teacher_calls = 0
        student_calls = 0
    elif optimizer == "BootstrapFewShot":
        # Teacher generates examples, student uses them
        teacher_calls = min(dataset_size * 3, 100)  # Cap at 100 calls
        student_calls = dataset_size  # One call per example for evaluation
    elif optimizer == "BootstrapRS":
        # More expensive due to random search
        teacher_calls = min(dataset_size * 10, 300)
        student_calls = dataset_size * 2
    elif optimizer == "MIPROv2":
        # Dataset-level optimization, not per-example
        teacher_calls = min(dataset_size + 100, 200)  # Bootstrap + optimization
        student_calls = dataset_size * 3  # Multiple evaluation rounds
    elif optimizer == "GEPA":
        # auto="light": a few hundred student rollouts, a handful of reflections
        teacher_calls = 30
        student_calls = max(dataset_size * 6, 400)
    elif optimizer == "COPRO":
        # Similar to MIPRO but focused on prompts
        teacher_calls = min(dataset_size + 50, 150)
        student_calls = dataset_size * 2
    elif optimizer == "BootstrapFinetune":
        # Similar to BootstrapFewShot + finetuning overhead
        teacher_calls = min(dataset_size * 3, 100)
        student_calls = dataset_size
        # Note: Finetuning costs are separate and not included here
    else:
        # Default conservative estimate
        teacher_calls = dataset_size * 5
        student_calls = dataset_size

    # Calculate costs (convert from per-1M to actual cost)
    teacher_input_cost = (teacher_calls * avg_input_tokens * teacher_pricing["input"]) / 1_000_000
    teacher_output_cost = (teacher_calls * avg_output_tokens * teacher_pricing["output"]) / 1_000_000
    student_input_cost = (student_calls * avg_input_tokens * student_pricing["input"]) / 1_000_000
    student_output_cost = (student_calls * avg_output_tokens * student_pricing["output"]) / 1_000_000

    total_cost = teacher_input_cost + teacher_output_cost + student_input_cost + student_output_cost

    # Add 20% buffer for safety
    total_cost_with_buffer = total_cost * 1.2

    return {
        "estimated_cost_usd": round(total_cost_with_buffer, 4),
        "teacher_calls": teacher_calls,
        "student_calls": student_calls,
        "estimated_input_tokens": int((teacher_calls + student_calls) * avg_input_tokens),
        "estimated_output_tokens": int((teacher_calls + student_calls) * avg_output_tokens),
        "student_model": student_model,
        "teacher_model": teacher_model,
        "optimizer": optimizer,
        "dataset_size": dataset_size,
        "is_expensive": total_cost_with_buffer > 1.0,
        "is_very_expensive": total_cost_with_buffer > 5.0,
        "breakdown": {
            "teacher_input": round(teacher_input_cost, 4),
            "teacher_output": round(teacher_output_cost, 4),
            "student_input": round(student_input_cost, 4),
            "student_output": round(student_output_cost, 4)
        }
    }


def calculate_actual_cost(
    student_model: str,
    teacher_model: str,
    lms: Optional[List[dspy.LM]] = None
) -> Dict[str, Any]:
    """
    Calculate actual cost from DSPy LM history.

    Every dspy.LM keeps a history of its calls, including the token usage and
    the cost reported for each one. This adds them up.

    Args:
        student_model: Model name used for the student program
        teacher_model: Model name used for teaching/optimization
        lms: The [student_lm, teacher_lm] objects used for the compilation.
             Defaults to the currently active DSPy LM.

    Returns:
        Dictionary containing:
        - actual_cost_usd: Total actual cost in USD
        - input_tokens: Total input tokens used
        - output_tokens: Total output tokens used
        - total_tokens: Sum of input and output tokens
        - breakdown: Dict with per-model cost breakdown
    """
    if lms is None:
        lms = [dspy.settings.lm] if dspy.settings.lm is not None else []

    pricing = [get_model_pricing(student_model), get_model_pricing(teacher_model)]
    roles = ["student", "teacher"]
    breakdown = {"student_input": 0.0, "student_output": 0.0, "teacher_input": 0.0, "teacher_output": 0.0}

    total_cost = 0.0
    total_input_tokens = 0
    total_output_tokens = 0

    seen = set()
    for index, lm in enumerate(lms):
        if lm is None or id(lm) in seen:
            continue
        seen.add(id(lm))
        role = roles[min(index, 1)]
        price = pricing[min(index, 1)]

        for call in getattr(lm, "history", []):
            usage = call.get("usage") or {}
            if not usage:
                # Answered from DSPy's cache: no tokens were used, nothing was charged
                continue
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens

            input_cost = (input_tokens * price["input"]) / 1_000_000
            output_cost = (output_tokens * price["output"]) / 1_000_000
            breakdown[f"{role}_input"] += input_cost
            breakdown[f"{role}_output"] += output_cost

            # Prefer the cost DSPy recorded for the call; fall back to the price list.
            cost = call.get("cost")
            total_cost += cost if cost is not None else input_cost + output_cost

    return {
        "actual_cost_usd": round(total_cost, 4),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "breakdown": {key: round(value, 4) for key, value in breakdown.items()}
    }


# -----------------------------------------------------------------------------
# Running a saved program (the "Generate" button)
# -----------------------------------------------------------------------------

# function to take a program from the program folder and run it on a row from the dataset
def generate_program_response(
    human_readable_id: str,
    row_data: Dict[str, Any],
    evaluate: bool = False
) -> Union[str, Tuple[str, Optional[float], Optional[str]]]:
    """
    Generate a response from a compiled program for a given row of data.

    Args:
        human_readable_id: The ID of the compiled program
        row_data: Dictionary containing input and optionally output (gold) data
        evaluate: If True, evaluate the prediction against gold data (if available)

    Returns:
        If evaluate=False: output string with prediction
        If evaluate=True: tuple of (output_string, evaluation_score, metric_type)
    """
    # Load the program details
    program_path = f"programs/{human_readable_id}.json"
    prompt_path = f"prompts/{human_readable_id}.json"

    print("program_path:", program_path)

    if not os.path.exists(program_path):
        raise ValueError(f"Compiled program not found: {program_path}")

    with open(prompt_path, 'r') as f:
        program_details = json.load(f)

    # Extract necessary information from program details
    input_fields = program_details.get('input_fields', [])
    input_descs = program_details.get('input_descs', [])
    output_fields = program_details.get('output_fields', [])
    output_descs = program_details.get('output_descs', [])
    dspy_module = program_details.get('dspy_module', 'Predict')
    instructions = program_details.get('instructions', '')
    metric_type = program_details.get('metric_type', 'Exact Match')
    judge_prompt_id = program_details.get('judge_prompt_id')

    # Get optional parameters
    hint = program_details.get('hint')
    max_iters = program_details.get('max_iters', 3)

    # Re-create the Signature the program was compiled with (BOOK: Signatures)
    signature = create_custom_signature(input_fields, output_fields, instructions, input_descs, output_descs)

    program_input = {}
    for field in input_fields:
        if field in row_data:
            program_input[field] = str(row_data[field])
        else:
            print(f"Warning: Required input field '{field}' not found in row_data")
            program_input[field] = ""  # or some default value

    lm = _SESSION_LMS.get(human_readable_id) or make_lm(program_details.get('llm_model', DEFAULT_STUDENT_MODEL))

    with dspy.context(lm=lm):
        # Run the program (BOOK: Inference)
        try:
            result, _messages = run_inference(program_path, signature, dspy_module, program_input, hint=hint, max_iters=max_iters)
            print("result:", result)
        except Exception as e:
            print(f"Error executing program: {str(e)}")
            error_msg = f"Error: {str(e)}"
            if evaluate:
                return error_msg, None, None
            return error_msg

        # Prepare the output
        output = "Input:\n"
        for field in input_fields:
            output += f"{field}: {program_input[field]}\n"

        output += "\nOutput:\n"
        for field in output_fields:
            output += f"{field}: {getattr(result, field)}\n"

        # Optionally evaluate the prediction
        if evaluate:
            # Check if we have gold data for all output fields
            has_gold_data = all(field in row_data for field in output_fields)

            if has_gold_data:
                try:
                    score = evaluate_single_prediction(
                        row_data, result, metric_type, output_fields, judge_prompt_id
                    )
                    return output, score, metric_type
                except Exception as e:
                    print(f"Error during evaluation: {e}")
                    return output, None, metric_type
            else:
                # No gold data available
                return output, None, metric_type

    return output


def create_evaluation_metric(
    metric_type: str,
    output_fields: List[str],
    judge_prompt_id: Optional[str] = None
) -> Callable[..., Any]:
    """
    Create an evaluation metric function based on the specified type.

    Kept for backwards compatibility: this is create_metric() (see
    "BOOK: Optimization -- metrics") with the judge chosen by prompt ID.
    """
    metric: Callable[..., Any] = create_metric(metric_type, output_fields, judge_prompt_id=judge_prompt_id)
    return metric


def evaluate_single_prediction(
    gold_data: Dict[str, Any],
    prediction: Any,
    metric_type: str,
    output_fields: List[str],
    judge_prompt_id: Optional[str] = None
) -> float:
    """
    Evaluate a single prediction against gold data using the specified metric.

    Args:
        gold_data: Dictionary containing the expected outputs
        prediction: The prediction object (dspy.Prediction or dict)
        metric_type: Type of metric to use
        output_fields: List of output field names
        judge_prompt_id: ID of judge prompt (for LLM-as-a-Judge)

    Returns:
        Numeric evaluation score
    """
    metric = create_metric(metric_type, output_fields, judge_prompt_id=judge_prompt_id)
    score = metric(gold_data, prediction, None)  # Pass None for trace parameter
    return float(score)


# -----------------------------------------------------------------------------
# Consolidated .dspyui files (one file holding the config, program and dataset)
# -----------------------------------------------------------------------------

def save_consolidated_program(
    human_readable_id: str,
    prompt_config: Dict[str, Any],
    compiled_program_path: str,
    dataset: pd.DataFrame
) -> str:
    """
    Save all program data to a single consolidated .dspyui file.

    Args:
        human_readable_id: Unique identifier for the program
        prompt_config: Dictionary containing all prompt configuration
        compiled_program_path: Path to the compiled DSPy program JSON
        dataset: DataFrame containing the training data

    Returns:
        Path to the saved .dspyui file
    """
    # Create consolidated_programs directory if it doesn't exist
    os.makedirs("consolidated_programs", exist_ok=True)

    # Load the compiled program
    with open(compiled_program_path, 'r') as f:
        compiled_program_data = json.load(f)

    # Create consolidated structure
    consolidated_data = {
        "version": "1.0",
        "human_readable_id": human_readable_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "dspy_version": dspy.__version__,
            "dspyui_version": DSPYUI_VERSION
        },
        "prompt_config": prompt_config,
        "compiled_program": compiled_program_data,
        "dataset": {
            "columns": list(dataset.columns),
            "data": json.loads(dataset.to_json(orient="records"))
        }
    }

    # Save to file
    output_path = f"consolidated_programs/{human_readable_id}.dspyui"
    with open(output_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return output_path


def load_consolidated_program(
    filepath: str
) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    """
    Load program data from a consolidated .dspyui file.

    Args:
        filepath: Path to the .dspyui file

    Returns:
        Tuple of (prompt_config, compiled_program_data, dataset)
    """
    with open(filepath, 'r') as f:
        consolidated_data = json.load(f)

    # Validate version
    if consolidated_data.get("version") != "1.0":
        raise ValueError(f"Unsupported .dspyui version: {consolidated_data.get('version')}")

    # Extract components
    prompt_config = consolidated_data["prompt_config"]
    compiled_program_data = consolidated_data["compiled_program"]
    dataset = pd.DataFrame(
        data=consolidated_data["dataset"]["data"],
        columns=consolidated_data["dataset"]["columns"]
    )

    return prompt_config, compiled_program_data, dataset


def export_to_consolidated(human_readable_id: str) -> Optional[str]:
    """
    Export an existing program (saved in old 3-file format) to consolidated .dspyui format.

    Args:
        human_readable_id: ID of the program to export

    Returns:
        Path to the created .dspyui file, or None if any files are missing
    """
    # Check if all required files exist
    prompt_path = f"prompts/{human_readable_id}.json"
    program_path = f"programs/{human_readable_id}.json"
    dataset_path = f"datasets/{human_readable_id}.csv"

    if not all(os.path.exists(p) for p in [prompt_path, program_path, dataset_path]):
        return None

    # Load the data
    with open(prompt_path, 'r') as f:
        prompt_config = json.load(f)

    dataset = pd.read_csv(dataset_path)

    # Save as consolidated file
    output_path = save_consolidated_program(
        human_readable_id,
        prompt_config,
        program_path,
        dataset
    )

    return output_path


def import_consolidated_program(filepath: str) -> Tuple[bool, str, str]:
    """
    Import a .dspyui file and save its components to the standard locations.

    Args:
        filepath: Path to the .dspyui file to import

    Returns:
        Tuple of (success: bool, message: str, human_readable_id: str)
    """
    try:
        # Load the consolidated file
        prompt_config, compiled_program_data, dataset = load_consolidated_program(filepath)

        # Get the human_readable_id
        human_readable_id = prompt_config.get('human_readable_id')
        if not human_readable_id:
            return False, "Error: No human_readable_id found in .dspyui file", ""

        # The ID becomes a file name, so it must not point anywhere else
        human_readable_id = os.path.basename(str(human_readable_id).replace("\\", "/"))
        if os.name == "nt":
            human_readable_id = human_readable_id.replace(":", "-")
        prompt_config['human_readable_id'] = human_readable_id

        # Create directories if they don't exist
        os.makedirs("prompts", exist_ok=True)
        os.makedirs("programs", exist_ok=True)
        os.makedirs("datasets", exist_ok=True)

        # Save prompt config
        prompt_path = f"prompts/{human_readable_id}.json"
        with open(prompt_path, 'w') as f:
            json.dump(prompt_config, f, indent=2)

        # Save compiled program
        program_path = f"programs/{human_readable_id}.json"
        with open(program_path, 'w') as f:
            json.dump(compiled_program_data, f, indent=2)

        # Save dataset
        dataset_path = f"datasets/{human_readable_id}.csv"
        dataset.to_csv(dataset_path, index=False)

        success_msg = f"✅ Successfully imported program '{human_readable_id}'\n\n"
        success_msg += f"- Prompt config: `{prompt_path}`\n"
        success_msg += f"- Compiled program: `{program_path}`\n"
        success_msg += f"- Dataset: `{dataset_path}`"

        return True, success_msg, human_readable_id

    except Exception as e:
        import traceback
        error_msg = f"❌ Import failed: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return False, error_msg, ""


# ========================================
# LM Configuration Helper
# ========================================

def configure_lm(
    model: str,
    provider: str = "openai",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None
) -> dspy.LM:
    """
    Create the LM for a model and try to make it DSPy's default.

    Args:
        model: Model name (e.g., 'openai/gpt-5.6-luna', 'local:qwen')
        provider: Unused; kept for backwards compatibility (the provider is
            part of the model name)
        api_key: Optional API key (uses environment variable if not provided)
        base_url: Optional base URL for local LLMs

    Returns:
        The dspy.LM. Inside the web app use `with dspy.context(lm=lm)`, because
        dspy.configure() only works from the thread that first called it.
    """
    lm = make_lm(model, base_url, **({"api_key": api_key} if api_key else {}))
    try:
        dspy.configure(lm=lm)
    except RuntimeError:
        pass  # called from a worker thread; the caller uses dspy.context(lm=lm)
    return lm


# ========================================
# Chain Building Functions
# ========================================

class ChainedProgram(dspy.Module):
    """
    A DSPy module that chains multiple modules together sequentially.
    Output from each step becomes input to the next step based on field mapping.
    """

    def __init__(self, steps: List[Dict[str, Any]]):
        """
        Initialize a chained program.

        Args:
            steps: List of step configurations, each containing:
                - module: The DSPy module instance
                - input_mapping: Dict mapping chain inputs/previous outputs to module inputs
                - output_fields: List of output field names from this step
        """
        super().__init__()
        self.steps = steps

        # Store modules as attributes for DSPy to track them
        for i, step in enumerate(steps):
            setattr(self, f"step_{i}", step["module"])

    def forward(self, **inputs):
        """Execute the chain sequentially."""
        # Start with initial inputs
        context = dict(inputs)

        # Execute each step
        for i, step in enumerate(self.steps):
            # Use the tracked attribute, so optimizers that copy the program
            # run (and improve) the copy's modules.
            module = getattr(self, f"step_{i}")
            input_mapping = step["input_mapping"]

            # Prepare inputs for this step based on mapping
            step_inputs = {}
            for module_input, source in input_mapping.items():
                if source in context:
                    step_inputs[module_input] = context[source]
                else:
                    raise ValueError(f"Step {i}: Cannot find '{source}' in context. Available: {list(context.keys())}")

            # Execute this step
            result = module(**step_inputs)

            # Add outputs to context
            for field in step["output_fields"]:
                if hasattr(result, field):
                    context[field] = getattr(result, field)

        # Return final prediction with all accumulated context
        return dspy.Prediction(**context)


def create_chain_step(
    module_type: str,
    signature_str: str,
    instructions: str = ""
) -> dspy.Module:
    """
    Create a single step (module) for a chain.

    Args:
        module_type: Type of module ('Predict', 'ChainOfThought', etc.)
        signature_str: Signature string (e.g., 'input -> output')
        instructions: Optional instructions for the module

    Returns:
        Configured DSPy module
    """
    # Create signature with instructions if provided
    signature = dspy.Signature(signature_str, instructions) if instructions else dspy.Signature(signature_str)

    # Create module based on type
    if module_type == "Predict":
        return dspy.Predict(signature)
    elif module_type in ("ChainOfThought", "ChainOfThoughtWithHint"):
        # Chain steps have no hint box, so WithHint behaves as ChainOfThought.
        return dspy.ChainOfThought(signature)
    elif module_type == "ProgramOfThought":
        return dspy.ProgramOfThought(signature)
    else:
        raise ValueError(f"Unsupported module type: {module_type}")


def _build_chain(chain_config: Dict[str, Any]) -> ChainedProgram:
    chain_steps = []
    for step_config in chain_config["steps"]:
        module = create_chain_step(
            step_config["module_type"],
            step_config["signature"],
            step_config.get("instructions", "")
        )

        # Parse output fields from signature
        sig_parts = step_config["signature"].split("->")
        if len(sig_parts) != 2:
            raise ValueError(f"Invalid signature: {step_config['signature']}")
        output_fields = [f.strip() for f in sig_parts[1].split(",") if f.strip()]

        chain_steps.append({
            "module": module,
            "input_mapping": step_config["input_mapping"],
            "output_fields": output_fields
        })

    return ChainedProgram(chain_steps)


def compile_chain(
    chain_config: Dict[str, Any],
    dataset: pd.DataFrame,
    model: str,
    optimizer_name: str = "BootstrapFewShot",
    metric_type: str = "Exact Match",
    judge_prompt_id: Optional[str] = None,
    provider: str = "openai",
    api_key: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    max_bootstrapped_demos: int = 8,
    max_labeled_demos: int = 8
) -> Tuple[ChainedProgram, Dict[str, Any], Dict[str, Any]]:
    """
    Compile a chained program using DSPy optimizers.

    Args:
        chain_config: Configuration dict containing:
            - name: Name of the chain
            - description: Description
            - initial_inputs: List of input field names for the entire chain
            - final_outputs: List of output field names from the final step
            - steps: List of step configs with module_type, signature, instructions, input_mapping
        dataset: Training data
        model: LLM model name
        optimizer_name: Name of DSPy optimizer
        metric_type: Evaluation metric type
        judge_prompt_id: Optional judge prompt ID for LLM-as-a-Judge
        provider: Unused; kept for backwards compatibility
        api_key: API key (uses environment variable if not provided)
        llm_base_url: Base URL for local LLMs
        max_bootstrapped_demos: Max examples for bootstrap optimizers
        max_labeled_demos: Max examples for labeled optimizers

    Returns:
        Tuple of (compiled_chain, evaluation_results, cost_data)
    """
    optimizer_name = normalize_optimizer_name(optimizer_name)
    lm = make_lm(model, llm_base_url, **({"api_key": api_key} if api_key else {}))

    # Create chained program
    chain_program = _build_chain(chain_config)

    # Prepare training data
    trainset = []
    for _, row in dataset.dropna(how="all").iterrows():
        example_dict = {key: str(value) for key, value in row.to_dict().items()}
        trainset.append(dspy.Example(**example_dict).with_inputs(*chain_config["initial_inputs"]))

    # Split into train/dev sets
    split_point = int(len(trainset) * 0.8)
    train_examples = trainset[:split_point]
    dev_examples = trainset[split_point:] if split_point < len(trainset) else trainset[:2]

    final_outputs = chain_config["final_outputs"]

    with dspy.context(lm=lm):
        # Create evaluation metric
        # (The chain tab has no Judge Prompt selector, so LLM-as-a-Judge falls
        # back to the generic reference-vs-candidate judge on the same model.)
        metric = create_metric(metric_type, final_outputs, judge_model=lm, judge_prompt_id=judge_prompt_id)

        # Configure and run optimizer
        if optimizer_name == "LabeledFewShot":
            optimizer = dspy.LabeledFewShot(k=min(max_labeled_demos, len(train_examples)))
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
        elif optimizer_name == "BootstrapFewShot":
            optimizer = dspy.BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=min(max_bootstrapped_demos, len(train_examples))
            )
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
        elif optimizer_name == "BootstrapRS":
            optimizer = dspy.BootstrapFewShotWithRandomSearch(
                metric=metric,
                max_bootstrapped_demos=min(max_bootstrapped_demos, len(train_examples)),
                num_candidate_programs=3
            )
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
        elif optimizer_name == "MIPROv2":
            optimizer = dspy.MIPROv2(
                metric=metric,
                auto="light",
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos
            )
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples, valset=dev_examples)
        elif optimizer_name == "GEPA":
            optimizer = dspy.GEPA(metric=with_feedback(metric, final_outputs), auto="light", reflection_lm=lm)
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples, valset=dev_examples)
        elif optimizer_name == "COPRO":
            optimizer = dspy.COPRO(metric=metric, breadth=3, depth=1)
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples, eval_kwargs={})
        elif optimizer_name == "BootstrapFinetune":
            optimizer = dspy.BootstrapFinetune(metric=metric)
            compiled_chain = optimizer.compile(chain_program, trainset=train_examples)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Evaluate on dev set (a percentage from 0 to 100)
        evaluator = dspy.Evaluate(devset=dev_examples, metric=metric, num_threads=1, display_progress=True)
        score = evaluator(compiled_chain).score

    # Calculate cost
    cost_data = calculate_actual_cost(model, model, lms=[lm])

    evaluation_results = {
        "score": score,
        "dev_set_size": len(dev_examples),
        "train_set_size": len(train_examples)
    }

    return compiled_chain, evaluation_results, cost_data


def save_chain_program(
    human_readable_id: str,
    chain_config: Dict[str, Any],
    compiled_chain: ChainedProgram,
    dataset: pd.DataFrame,
    evaluation_results: Dict[str, Any],
    cost_data: Dict[str, Any],
    model: str,
    optimizer_name: str,
    metric_type: str
) -> str:
    """
    Save a compiled chain program and its configuration.

    Args:
        human_readable_id: Unique ID for the chain
        chain_config: Chain configuration dict
        compiled_chain: Compiled ChainedProgram
        dataset: Training dataset
        evaluation_results: Evaluation results dict
        cost_data: Cost tracking data
        model: Model name
        optimizer_name: Optimizer name
        metric_type: Metric type

    Returns:
        Path to saved consolidated file
    """
    # Create directories
    os.makedirs("chains", exist_ok=True)
    os.makedirs("consolidated_programs", exist_ok=True)

    # Save chain program
    chain_path = f"chains/{human_readable_id}.json"
    compiled_chain.save(chain_path)

    # Create metadata
    metadata = {
        "human_readable_id": human_readable_id,
        "chain_name": chain_config["name"],
        "description": chain_config.get("description", ""),
        "model": model,
        "optimizer": optimizer_name,
        "metric": metric_type,
        "evaluation_score": evaluation_results["score"],
        "train_set_size": evaluation_results["train_set_size"],
        "dev_set_size": evaluation_results["dev_set_size"],
        "actual_cost": cost_data.get("actual_cost_usd", 0),
        "total_tokens": cost_data.get("total_tokens", 0),
        "input_tokens": cost_data.get("input_tokens", 0),
        "output_tokens": cost_data.get("output_tokens", 0),
        "created_at": datetime.datetime.now().isoformat(),
        "chain_config": chain_config
    }

    # Save metadata
    metadata_path = f"chains/{human_readable_id}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save consolidated file
    with open(chain_path, 'r') as f:
        compiled_chain_data = json.load(f)

    consolidated_data = {
        "version": "1.0",
        "type": "chain",
        "human_readable_id": human_readable_id,
        "metadata": {
            "created_at": datetime.datetime.now().isoformat(),
            "dspy_version": dspy.__version__,
            "dspyui_version": DSPYUI_VERSION
        },
        "chain_config": chain_config,
        "model_config": {
            "model": model,
            "optimizer": optimizer_name,
            "metric": metric_type
        },
        "evaluation_results": evaluation_results,
        "cost_data": cost_data,
        "compiled_program": compiled_chain_data,
        "dataset": {
            "columns": list(dataset.columns),
            "data": json.loads(dataset.to_json(orient="records"))
        }
    }

    consolidated_path = f"consolidated_programs/{human_readable_id}.dspyui"
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2)

    return consolidated_path


def execute_chain(
    human_readable_id: str,
    inputs: Dict[str, Any]
) -> Tuple[dspy.Prediction, Optional[float]]:
    """
    Execute a saved chain program on new inputs.

    Args:
        human_readable_id: ID of the saved chain
        inputs: Dict of input values

    Returns:
        Tuple of (prediction, evaluation_score)
    """
    # Load chain
    chain_path = f"chains/{human_readable_id}.json"
    metadata_path = f"chains/{human_readable_id}_metadata.json"

    if not os.path.exists(chain_path):
        raise FileNotFoundError(f"Chain not found: {chain_path}")

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Create chain structure from config, then load the compiled state
    chain_program = _build_chain(metadata["chain_config"])
    chain_program.load(chain_path)

    # Execute with the model the chain was compiled for
    with dspy.context(lm=make_lm(metadata.get("model", DEFAULT_STUDENT_MODEL))):
        prediction = chain_program(**inputs)

    return prediction, None


def list_chains(
    name_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all saved chain programs.

    Args:
        name_filter: Optional filter string for chain names

    Returns:
        List of chain metadata dicts
    """
    if not os.path.exists('chains'):
        return []

    chains = []
    for file in os.listdir('chains'):
        if file.endswith('_metadata.json'):
            with open(os.path.join('chains', file), 'r') as f:
                metadata = json.load(f)

                if name_filter and name_filter.lower() not in metadata["chain_name"].lower():
                    continue

                chains.append({
                    "ID": metadata["human_readable_id"],
                    "Name": metadata["chain_name"],
                    "Description": metadata.get("description", ""),
                    "Steps": len(metadata["chain_config"]["steps"]),
                    "Eval Score": metadata.get("evaluation_score", "N/A"),
                    "Cost": metadata.get("actual_cost", 0),
                    "Tokens": metadata.get("total_tokens", 0),
                    "Model": metadata.get("model", ""),
                    "Details": json.dumps(metadata, indent=2)
                })

    return chains


# Older name for create_module(), kept so existing scripts keep working.
create_dspy_module = create_module
