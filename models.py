"""
Pydantic models for DSPyUI data validation.

This module provides type-safe data models for validating user inputs,
program configurations, and compilation parameters.
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import pandas as pd


class FieldDefinition(BaseModel):
    """Represents a single input or output field definition."""

    name: str = Field(..., min_length=1, description="Field name (required)")
    description: str = Field(default="", description="Optional field description")

    model_config = ConfigDict(frozen=False)

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that field name contains only valid characters."""
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Field name must contain only alphanumeric characters, underscores, and hyphens")
        return v


class SignatureDefinition(BaseModel):
    """Represents the complete signature for a DSPy program."""

    input_fields: List[FieldDefinition] = Field(..., min_length=1)
    output_fields: List[FieldDefinition] = Field(..., min_length=1)
    instructions: str = Field(..., min_length=1, description="Task instructions")

    model_config = ConfigDict(frozen=False)

    def get_input_names(self) -> List[str]:
        """Get list of input field names."""
        return [field.name for field in self.input_fields]

    def get_output_names(self) -> List[str]:
        """Get list of output field names."""
        return [field.name for field in self.output_fields]

    def get_input_descriptions(self) -> List[str]:
        """Get list of input field descriptions."""
        return [field.description for field in self.input_fields]

    def get_output_descriptions(self) -> List[str]:
        """Get list of output field descriptions."""
        return [field.description for field in self.output_fields]


class ModelConfig(BaseModel):
    """Configuration for LLM models."""

    student_model: str = Field(..., description="Model used for the student program")
    teacher_model: str = Field(..., description="Model used for teaching/optimization")
    student_base_url: Optional[str] = Field(None, description="Base URL for local student model")
    teacher_base_url: Optional[str] = Field(None, description="Base URL for local teacher model")

    model_config = ConfigDict(frozen=False)

    @field_validator('student_model', 'teacher_model')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty."""
        if not v.strip():
            raise ValueError("Model name cannot be empty")
        return v


class OptimizerConfig(BaseModel):
    """Configuration for DSPy optimizers."""

    optimizer_type: Literal[
        "BootstrapFewShot",
        "BootstrapFewShotWithRandomSearch",
        "COPRO",
        "MIPROv2",
        "LabeledFewShot",
        "BootstrapFinetune"
    ] = Field(..., description="Type of optimizer to use")

    k: int = Field(default=16, ge=1, le=50, description="Number of examples for LabeledFewShot")

    model_config = ConfigDict(frozen=False)


class ModuleConfig(BaseModel):
    """Configuration for DSPy modules."""

    module_type: Literal[
        "Predict",
        "ChainOfThought",
        "ChainOfThoughtWithHint",
        "ProgramOfThought"
    ] = Field(..., description="Type of DSPy module to use")

    hint: Optional[str] = Field(None, description="Hint for ChainOfThoughtWithHint")
    max_iters: int = Field(default=3, ge=1, le=10, description="Max iterations for ProgramOfThought")

    model_config = ConfigDict(frozen=False)

    @field_validator('hint')
    @classmethod
    def validate_hint(cls, v: Optional[str], info) -> Optional[str]:
        """Validate hint is provided when using ChainOfThoughtWithHint."""
        # Note: We can't access other fields directly in Pydantic v2,
        # so we'll validate this at the ProgramConfig level
        return v


class MetricConfig(BaseModel):
    """Configuration for evaluation metrics."""

    metric_type: Literal[
        "Exact Match",
        "Cosine Similarity",
        "LLM-as-a-Judge"
    ] = Field(..., description="Type of evaluation metric")

    judge_prompt_id: Optional[str] = Field(None, description="ID of judge prompt (for LLM-as-a-Judge)")

    model_config = ConfigDict(frozen=False)

    @field_validator('judge_prompt_id')
    @classmethod
    def validate_judge_prompt(cls, v: Optional[str], info) -> Optional[str]:
        """Validate judge_prompt_id is provided when using LLM-as-a-Judge."""
        # This will be validated at ProgramConfig level
        return v


class ProgramConfig(BaseModel):
    """Complete configuration for compiling a DSPy program."""

    signature: SignatureDefinition
    model_config_data: ModelConfig = Field(..., alias="model_config")
    optimizer: OptimizerConfig
    module: ModuleConfig
    metric: MetricConfig

    model_config = ConfigDict(frozen=False, populate_by_name=True)

    @field_validator('module')
    @classmethod
    def validate_module_hint(cls, v: ModuleConfig) -> ModuleConfig:
        """Validate that hint is provided for ChainOfThoughtWithHint."""
        if v.module_type == "ChainOfThoughtWithHint" and not v.hint:
            raise ValueError("Hint is required for ChainOfThoughtWithHint module")
        return v

    @field_validator('metric')
    @classmethod
    def validate_metric_judge(cls, v: MetricConfig) -> MetricConfig:
        """Validate that judge_prompt_id is provided for LLM-as-a-Judge."""
        if v.metric_type == "LLM-as-a-Judge" and not v.judge_prompt_id:
            raise ValueError("Judge prompt ID is required for LLM-as-a-Judge metric")
        return v


class ExampleData(BaseModel):
    """Represents example training data."""

    data: List[Dict[str, Any]] = Field(..., min_length=2, description="At least 2 examples required")

    model_config = ConfigDict(frozen=False)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame(self.data)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "ExampleData":
        """Create from pandas DataFrame."""
        return cls(data=df.to_dict('records'))

    @field_validator('data')
    @classmethod
    def validate_min_examples(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate minimum number of examples."""
        if len(v) < 2:
            raise ValueError("At least 2 examples are required for compilation")
        return v


class PromptDetails(BaseModel):
    """Represents saved prompt configuration details."""

    human_readable_id: str
    input_fields: List[str]
    output_fields: List[str]
    input_descs: List[str] = Field(default_factory=list)
    output_descs: List[str] = Field(default_factory=list)
    instructions: str
    dspy_module: str
    llm_model: str
    teacher_model: str
    optimizer: str
    metric_type: str
    evaluation_score: Optional[float] = None
    baseline_score: Optional[float] = None
    judge_prompt_id: Optional[str] = None
    hint: Optional[str] = None
    max_iters: int = 3
    k: int = 16

    model_config = ConfigDict(frozen=False)

    @classmethod
    def from_json_file(cls, filepath: str) -> "PromptDetails":
        """Load prompt details from JSON file."""
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, filepath: str) -> None:
        """Save prompt details to JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)


class CompilationResult(BaseModel):
    """Result of program compilation."""

    success: bool
    message: str
    human_readable_id: Optional[str] = None
    evaluation_score: Optional[float] = None
    baseline_score: Optional[float] = None
    final_prompt: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(frozen=False)


class GenerationResult(BaseModel):
    """Result of program generation (inference)."""

    output: str
    evaluation_score: Optional[float] = None
    metric_type: Optional[str] = None
    error: Optional[str] = None

    model_config = ConfigDict(frozen=False)


# Supported model lists
SUPPORTED_OPENAI_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo"
]

SUPPORTED_ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307"
]

SUPPORTED_GROQ_MODELS = [
    "mixtral-8x7b-32768",
    "gemma-7b-it",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "gemma2-9b-it"
]

SUPPORTED_GOOGLE_MODELS = [
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

ALL_SUPPORTED_MODELS = (
    SUPPORTED_OPENAI_MODELS +
    SUPPORTED_ANTHROPIC_MODELS +
    SUPPORTED_GROQ_MODELS +
    SUPPORTED_GOOGLE_MODELS +
    ["local:model-name"]  # Placeholder for local models
)
