"""
Offline unit tests for dspyui.core (no API calls, no browser).

Run with:  pytest tests/test_core.py
"""
import json

import dspy
import pandas as pd
import pytest

from dspyui import core


RATING_FIELDS = (["joke", "topic"], ["funny"])


def make_rating_signature():
    return core.create_custom_signature(
        ["joke", "topic"],
        ["funny"],
        "Rate whether a joke is funny",
        ["The joke to be rated", ""],
        ["Whether the joke is funny or not, 1 or 0."],
    )


class TestSignatures:
    def test_fields_and_instructions(self):
        signature = make_rating_signature()
        assert list(signature.input_fields) == ["joke", "topic"]
        assert list(signature.output_fields) == ["funny"]
        assert signature.instructions == "Rate whether a joke is funny"
        assert signature.input_fields["joke"].json_schema_extra["desc"] == "The joke to be rated"

    def test_blank_description_is_not_rendered_as_none(self):
        signature = make_rating_signature()
        messages = dspy.ChatAdapter().format(signature, [], {"joke": "j", "topic": "t"})
        assert "None" not in messages[0]["content"]


class TestModules:
    def test_book_model_options_come_first(self):
        assert core.MODEL_OPTIONS[:4] == [
            "openai/gpt-5.6-luna",
            "openai/gpt-5.6-sol",
            "anthropic/claude-opus-4.8",
            "gemini/gemini-3.5-flash",
        ]
        assert core.DEFAULT_STUDENT_MODEL == "openai/gpt-5.6-luna"
        assert core.DEFAULT_TEACHER_MODEL == "openai/gpt-5.6-sol"

    @pytest.mark.parametrize("name,expected", [
        ("Predict", dspy.Predict),
        ("ChainOfThought", dspy.ChainOfThought),
        ("ChainOfThoughtWithHint", core.ChainOfThoughtWithHint),
        ("ProgramOfThought", dspy.ProgramOfThought),
    ])
    def test_create_module(self, name, expected):
        module = core.create_module(name, make_rating_signature(), hint="be strict")
        assert isinstance(module, expected)
        assert isinstance(module, dspy.Module)

    def test_unknown_module(self):
        with pytest.raises(ValueError):
            core.create_module("ReAct", make_rating_signature())

    @pytest.mark.parametrize("given,expected", [
        ("openai/gpt-5.6-luna", "openai/gpt-5.6-luna"),
        ("anthropic/claude-opus-4.8", "anthropic/claude-opus-4-8"),
        ("gemini/gemini-3.5-flash", "gemini/gemini-3.5-flash"),
        ("openrouter/anthropic/claude-opus-4.8", "openrouter/anthropic/claude-opus-4.8"),
        # names saved by older versions of DSPyUI
        ("gpt-4o-mini", "openai/gpt-4o-mini"),
        ("claude-3-haiku-20240307", "anthropic/claude-3-haiku-20240307"),
        ("gemini-1.5-flash", "gemini/gemini-1.5-flash"),
        ("llama3-70b-8192", "groq/llama3-70b-8192"),
    ])
    def test_normalize_model_name(self, given, expected):
        assert core.normalize_model_name(given) == expected

    def test_openrouter_fallback_when_provider_key_missing(self, monkeypatch):
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert core.make_lm("openai/gpt-5.6-luna").model == "openrouter/openai/gpt-5.6-luna"
        assert core.make_lm("anthropic/claude-opus-4.8").model == "openrouter/anthropic/claude-opus-4.8"
        assert core.make_lm("gemini/gemini-3.5-flash").model == "openrouter/google/gemini-3.5-flash"

    def test_one_openrouter_key_covers_every_model_in_the_lists(self, monkeypatch):
        for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
            monkeypatch.delenv(env, raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        for model in core.MODEL_OPTIONS:
            if not model.startswith("local:"):
                assert core.make_lm(model).model.startswith("openrouter/"), model
        assert core.make_lm("groq/openai/gpt-oss-120b").model == "openrouter/openai/gpt-oss-120b"

    def test_unfilled_placeholder_keys_count_as_missing(self, monkeypatch):
        # someone copied .env.example and only filled in the OpenRouter key
        monkeypatch.setenv("OPENAI_API_KEY", "fillthisin")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert core.make_lm("openai/gpt-5.6-luna").model == "openrouter/openai/gpt-5.6-luna"

    def test_direct_provider_when_key_present(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        assert core.make_lm("openai/gpt-5.6-luna").model == "openai/gpt-5.6-luna"

    def test_local_model(self):
        lm = core.make_lm("local:qwen", "http://127.0.0.1:1234/v1")
        assert lm.model == "openai/qwen"
        assert lm.kwargs["api_base"] == "http://127.0.0.1:1234/v1"


class TestDatasets:
    def test_split_is_80_20_and_strings(self):
        df = pd.read_csv(core.EXAMPLE_DATA_DIR / "rating_jokes.csv")
        trainset, devset = core.load_and_split(df, *RATING_FIELDS)
        assert len(trainset) == int(0.8 * len(df))
        assert len(trainset) + len(devset) == len(df)
        assert trainset[0].funny in ("0", "1")
        assert set(trainset[0].inputs().keys()) == {"joke", "topic"}

    def test_load_from_csv_path(self):
        trainset, devset = core.load_and_split(core.EXAMPLE_DATA_DIR / "rewriting_jokes.csv", ["joke", "comedian"], ["rewritten_joke"])
        assert len(trainset) >= 1 and len(devset) >= 1

    def test_two_rows_gives_one_each(self):
        df = pd.DataFrame({"joke": ["a", "b"], "topic": ["x", "y"], "funny": [1, 0]})
        trainset, devset = core.load_and_split(df, *RATING_FIELDS)
        assert (len(trainset), len(devset)) == (1, 1)

    def test_too_few_rows(self):
        df = pd.DataFrame({"joke": ["a"], "topic": ["x"], "funny": [1]})
        with pytest.raises(ValueError):
            core.load_and_split(df, *RATING_FIELDS)

    def test_missing_column(self):
        df = pd.DataFrame({"joke": ["a", "b"], "funny": [1, 0]})
        with pytest.raises(ValueError, match="topic"):
            core.load_and_split(df, *RATING_FIELDS)

    def test_blank_rows_are_skipped(self):
        df = pd.DataFrame({"joke": ["a", "", "b", None], "topic": ["x", " ", "y", None], "funny": ["1", "", "0", None]})
        trainset, devset = core.load_and_split(df, *RATING_FIELDS)
        assert len(trainset) + len(devset) == 2

    def test_manual_rows_get_headers(self):
        rows = pd.DataFrame([["a", "x", "1"], ["b", "y", "0"], [None, None, None]])
        trainset, devset = core.load_manual_and_split(rows, *RATING_FIELDS)
        assert len(trainset) + len(devset) == 2


class TestMetrics:
    def test_exact_match_takes_five_arguments(self):
        metric = core.create_metric("Exact Match", ["funny"])
        gold = dspy.Example(joke="j", topic="t", funny="1").with_inputs("joke", "topic")
        assert metric(gold, dspy.Prediction(funny=" 1 "), None, None, None) == 1.0
        assert metric(gold, dspy.Prediction(funny="0")) == 0.0

    def test_exact_match_against_a_plain_row(self):
        # the "Generate" button compares against a row of the table (numbers, not strings)
        assert core.evaluate_single_prediction({"funny": 1}, dspy.Prediction(funny="1"), "Exact Match", ["funny"]) == 1.0

    def test_feedback_wrapper_for_gepa(self):
        metric = core.with_feedback(core.create_metric("Exact Match", ["funny"]), ["funny"])
        gold = dspy.Example(joke="j", topic="t", funny="1").with_inputs("joke", "topic")
        result = metric(gold, dspy.Prediction(funny="0"), None, "predict", None)
        assert result.score == 0.0
        assert '"funny": "1"' in result.feedback

    def test_judge_requires_a_prompt(self):
        with pytest.raises(ValueError):
            core.create_metric("LLM-as-a-Judge", ["joke"])

    def test_unknown_metric(self):
        with pytest.raises(ValueError):
            core.create_metric("BLEU", ["joke"])


class TestSavedPrograms:
    def test_bundled_judge_loads(self):
        # saved by DSPy 2.x; must still load, with its few-shot demos
        judge, inputs, outputs = core.load_judge_program(core.EXAMPLE_JUDGE_ID + ".json")
        assert (inputs, outputs) == (["joke", "topic"], ["funny"])
        demos = judge.predict.demos
        assert len(demos) > 0
        assert "rationale" not in demos[0] and "funny" in demos[0]

    def test_save_and_load_round_trip(self, tmp_path):
        signature = make_rating_signature()
        module = core.create_module("ChainOfThought", signature)
        module.predict.demos = [dspy.Example(joke="j", topic="t", reasoning="r", funny="1")]
        path = tmp_path / "program.json"
        module.save(path)

        fresh = core.load_program(core.create_module("ChainOfThought", signature), path)
        assert fresh.predict.demos[0]["funny"] == "1"

    def test_optimizer_names(self):
        assert core.OPTIMIZER_OPTIONS[:4] == ["BootstrapFewShot", "BootstrapRS", "MIPROv2", "GEPA"]
        assert core.normalize_optimizer_name("BootstrapFewShotWithRandomSearch") == "BootstrapRS"

    def test_human_readable_id_is_a_safe_file_name(self):
        program_id = core.generate_human_readable_id(
            ["joke", "topic"], ["funny"], "ChainOfThought",
            "openrouter/openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "BootstrapRS", "",
        )
        assert program_id.startswith("JokeTopic:Funny-Gpt56Luna_ChainOfThought_Bootstraprs-")
        assert "/" not in program_id and "." not in program_id

    def test_consolidated_round_trip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        program_path = tmp_path / "program.json"
        core.create_module("Predict", make_rating_signature()).save(program_path)
        dataset = pd.DataFrame({"joke": ["a", "b"], "topic": ["x", "y"], "funny": [1, 0]})
        config = {"human_readable_id": "Example-Id", "input_fields": ["joke", "topic"], "output_fields": ["funny"]}

        path = core.save_consolidated_program("Example-Id", config, str(program_path), dataset)
        ok, _message, program_id = core.import_consolidated_program(path)
        assert ok and program_id == "Example-Id"
        assert json.load(open("prompts/Example-Id.json"))["input_fields"] == ["joke", "topic"]
        assert len(pd.read_csv("datasets/Example-Id.csv")) == 2

    def test_pricing_lookup(self):
        assert core.get_model_pricing("local:qwen") == {"input": 0.0, "output": 0.0}
        price = core.get_model_pricing("openrouter/openai/gpt-5.6-luna")
        assert price["input"] > 0 and price["output"] > price["input"]


class TestApiKeys:
    """Keys typed into the Settings tab are per session and never touch os.environ."""

    PROVIDER_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY")

    @pytest.fixture
    def no_env_keys(self, monkeypatch):
        for env in self.PROVIDER_ENVS:
            monkeypatch.delenv(env, raising=False)

    def test_session_key_is_used_and_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        with core.use_api_keys({"OPENAI_API_KEY": "session-key"}):
            lm = core.make_lm("openai/gpt-5.6-luna")
        assert lm.model == "openai/gpt-5.6-luna"
        assert lm.kwargs["api_key"] == "session-key"

    def test_environment_keys_are_left_to_dspy(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        assert "api_key" not in core.make_lm("openai/gpt-5.6-luna").kwargs

    def test_session_keys_do_not_leak_out_of_the_block(self, no_env_keys):
        import os
        with core.use_api_keys({"OPENROUTER_API_KEY": "session-key"}):
            assert core._provider_key("openrouter") == "session-key"
            assert "OPENROUTER_API_KEY" not in os.environ
        assert core._provider_key("openrouter") is None
        assert "api_key" not in core.make_lm("openai/gpt-5.6-luna").kwargs

    def test_session_openrouter_key_enables_the_fallback(self, no_env_keys):
        with core.use_api_keys({"OPENROUTER_API_KEY": "session-key", "OPENAI_API_KEY": "  "}):
            lm = core.make_lm("anthropic/claude-opus-4.8")
        assert lm.model == "openrouter/anthropic/claude-opus-4.8"
        assert lm.kwargs["api_key"] == "session-key"

    def test_google_box_is_used_for_gemini(self, no_env_keys):
        with core.use_api_keys({"GOOGLE_API_KEY": "google-key"}):
            lm = core.make_lm("gemini/gemini-3.5-flash")
        assert lm.model == "gemini/gemini-3.5-flash"
        assert lm.kwargs["api_key"] == "google-key"

    def test_blank_or_missing_keys_are_ignored(self, no_env_keys):
        with core.use_api_keys({"OPENAI_API_KEY": "", "GROQ_API_KEY": None}):
            assert core._provider_key("openai") is None
        with core.use_api_keys(None):
            assert core._provider_key("openai") is None

    def test_key_is_not_written_into_saved_state(self):
        with core.use_api_keys({"OPENAI_API_KEY": "session-key"}):
            lm = core.make_lm("openai/gpt-5.6-luna")
        assert "session-key" not in json.dumps(lm.dump_state())

    def test_embedder_follows_the_session_keys(self, no_env_keys):
        with core.use_api_keys({"OPENROUTER_API_KEY": "session-key"}):
            embedder = core._make_embedder()
        assert embedder.model == "openrouter/openai/text-embedding-3-small"
        assert embedder.default_kwargs["api_key"] == "session-key"

    def test_environment_keys_are_never_sent_to_the_browser(self, tmp_path):
        """The page config is what every visitor's browser receives."""
        import os, subprocess, sys
        env = {**os.environ, **{name: f"SECRET-{name}" for name in self.PROVIDER_ENVS}}
        code = (
            "import json; from dspyui.interface import demo; "
            "print('CONFIG_HAS_SECRET' if 'SECRET-' in json.dumps(demo.get_config_file(), default=str) else 'CONFIG_CLEAN')"
        )
        result = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert "CONFIG_CLEAN" in result.stdout, result.stdout[-500:] + result.stderr[-1500:]


class TestLauncher:
    def test_launcher_rebuilds_the_interface_in_gradio_reload_mode(self, tmp_path):
        """`gradio interface.py` (used by webui.sh) resets Gradio's component counter and
        runs the launcher again. The counter must end up where it was after the first
        run, or components drawn later by gr.render collide with existing ones."""
        import os, subprocess, sys
        launcher = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interface.py")
        if not os.path.exists(launcher):
            pytest.skip("launcher is only present in a repository checkout")
        code = (
            "import sys\n"
            "from gradio.context import Context\n"
            f"source = open({launcher!r}).read().split('if __name__')[0]\n"
            "namespace = {'__name__': 'interface'}\n"
            "exec(source, namespace); first = Context.id\n"
            "Context.id = 0\n"
            "exec(source, namespace); second = Context.id\n"
            "print('COUNTERS', first, second)\n"
            "sys.exit(0 if first == second and first > 100 else 1)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout[-300:] + result.stderr[-1500:]
