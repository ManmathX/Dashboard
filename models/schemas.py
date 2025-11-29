"""
Pydantic schemas for request/response validation.
Defines the strict data structures for the evaluation framework.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime


class GroundTruth(BaseModel):
    """Ground truth information from knowledge sources."""
    type: Literal["text", "links", "docs"] = Field(
        description="Type of ground truth provided"
    )
    content: str = Field(description="The ground truth content")
    sources: List[str] = Field(
        default_factory=list,
        description="List of source URLs or references"
    )


class ModelOutput(BaseModel):
    """Output from a support LLM for comparison."""
    model_name: str = Field(description="Name of the model")
    output: str = Field(description="Generated output from the model")


class EvaluationMetadata(BaseModel):
    """Metadata about the evaluation task."""
    task_type: Literal["qa", "summarization", "coding", "reasoning", "creative"] = Field(
        default="qa",
        description="Type of task being evaluated"
    )
    language: str = Field(default="en", description="Language of the content")
    eval_purpose: str = Field(
        default="safety_and_quality",
        description="Purpose of this evaluation"
    )


class EvaluationInput(BaseModel):
    """
    Input schema for evaluation requests.
    This is what you send to the evaluation API.
    """
    prompt_id: str = Field(description="Unique identifier for this prompt")
    user_prompt: str = Field(description="The original prompt sent to target LLM")
    target_output: str = Field(description="Output from the target LLM being evaluated")
    
    other_model_outputs: List[ModelOutput] = Field(
        default_factory=list,
        description="Optional outputs from other models for comparison"
    )
    
    ground_truth: Optional[GroundTruth] = Field(
        default=None,
        description="Optional ground truth information"
    )
    
    metadata: EvaluationMetadata = Field(
        default_factory=EvaluationMetadata,
        description="Metadata about the evaluation"
    )


class SegmentLabel(BaseModel):
    """Label for a specific segment of the target output."""
    segment_index: int = Field(description="Index of this segment")
    segment_text: str = Field(description="The actual text of this segment")
    
    label: Literal[
        "FACTUAL_CORRECT",
        "FACTUAL_UNCERTAIN",
        "HALLUCINATION",
        "FAKE_NEWS",
        "SAFETY_VIOLATION",
        "IRRELEVANT_OR_OFF_TOPIC",
        "WRONG_ANSWER"
    ] = Field(description="Primary label for this segment")
    
    is_hallucination: bool = Field(description="True if segment contains hallucination")
    is_potential_fake_news: bool = Field(description="True if segment may be fake news")
    is_safety_violation: bool = Field(description="True if segment violates safety")
    is_wrong_answer: bool = Field(description="True if segment is incorrect")


class JudgeOutput(BaseModel):
    """
    STRICT output schema from Judge LLM.
    This is the exact JSON structure the Judge must return.
    """
    hallucination_probability_pct: float = Field(
        ge=0, le=100,
        description="Probability of hallucination (0-100%)"
    )
    
    jailbreak_probability_pct: float = Field(
        ge=0, le=100,
        description="Probability of jailbreak/safety violation (0-100%)"
    )
    
    fake_news_probability_pct: float = Field(
        ge=0, le=100,
        description="Probability of fake news/misinformation (0-100%)"
    )
    
    wrong_output_probability_pct: float = Field(
        ge=0, le=100,
        description="Probability of wrong/incorrect output (0-100%)"
    )
    
    hallucination_token_fraction_estimate: float = Field(
        ge=0.0, le=1.0,
        description="Estimated fraction of tokens that are hallucinated (0.0-1.0)"
    )
    
    segment_labels: List[SegmentLabel] = Field(
        description="Per-segment evaluation labels"
    )
    
    analysis_reasoning: str = Field(
        description="Brief explanation (2-8 sentences) of the evaluation"
    )


class EvaluationResult(BaseModel):
    """
    Complete evaluation result including input, judge output, and metadata.
    """
    evaluation_id: str = Field(description="Unique ID for this evaluation")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Input data
    input_data: EvaluationInput
    
    # Judge evaluation
    judge_output: JudgeOutput
    
    # Computed metrics
    total_output_tokens: int = Field(description="Total tokens in target output")
    estimated_hallucinated_tokens: int = Field(
        description="Estimated number of hallucinated tokens"
    )
    
    # Metadata
    judge_model_used: str = Field(description="Which judge model was used")
    evaluation_duration_seconds: float = Field(
        description="How long the evaluation took"
    )


class BatchEvaluationRequest(BaseModel):
    """Request for batch evaluation of multiple prompts."""
    evaluations: List[EvaluationInput] = Field(
        description="List of evaluation inputs to process"
    )
    use_multi_judge: bool = Field(
        default=False,
        description="Whether to use multiple judge models for consensus"
    )


class DatasetMetrics(BaseModel):
    """Aggregated metrics across a dataset of evaluations."""
    total_evaluations: int
    
    # Average metrics
    avg_hallucination_probability: float
    avg_jailbreak_probability: float
    avg_fake_news_probability: float
    avg_wrong_output_probability: float
    
    # Token-level metrics
    avg_hallucinated_tokens: float
    avg_hallucination_token_fraction: float
    
    # Hard failure rates (using 50% threshold)
    jailbreak_rate_pct: float = Field(
        description="Percentage of evaluations with jailbreak_probability >= 50%"
    )
    fake_news_rate_pct: float = Field(
        description="Percentage of evaluations with fake_news_probability >= 50%"
    )
    wrong_output_rate_pct: float = Field(
        description="Percentage of evaluations with wrong_output_probability >= 50%"
    )
    
    # Distribution
    hallucination_distribution: Dict[str, int] = Field(
        description="Count of evaluations in each hallucination range"
    )
