"""
Core evaluation orchestrator.
Coordinates the flow: Target LLM → Judge LLM → Results.
"""

import uuid
from datetime import datetime
from typing import Optional

from models.schemas import EvaluationInput, EvaluationResult, JudgeOutput
from llm.judge_llm import JudgeLLM
from llm.target_llm import TargetLLM


class Evaluator:
    """
    Main evaluation orchestrator.
    Coordinates target LLM testing and judge evaluation.
    """
    
    def __init__(self):
        """Initialize evaluator with judge LLM."""
        self.judge = JudgeLLM()
    
    async def evaluate(
        self,
        eval_input: EvaluationInput,
        target_llm: Optional[TargetLLM] = None
    ) -> EvaluationResult:
        """
        Perform complete evaluation of target LLM output.
        
        Args:
            eval_input: Evaluation input with prompt and target output
            target_llm: Optional TargetLLM instance for token counting
        
        Returns:
            Complete evaluation result
        """
        # Generate unique evaluation ID
        evaluation_id = str(uuid.uuid4())
        
        # Get judge evaluation
        judge_result = await self.judge.evaluate(eval_input)
        judge_output: JudgeOutput = judge_result["judge_output"]
        
        # Count tokens in target output
        if target_llm:
            total_tokens = target_llm.count_tokens(eval_input.target_output)
        else:
            # Use tiktoken directly for token counting (no API call needed)
            import tiktoken
            try:
                enc = tiktoken.encoding_for_model("gpt-4")
                total_tokens = len(enc.encode(eval_input.target_output))
            except Exception:
                # Fallback to cl100k_base encoding
                enc = tiktoken.get_encoding("cl100k_base")
                total_tokens = len(enc.encode(eval_input.target_output))
        
        # Calculate estimated hallucinated tokens
        estimated_hallucinated_tokens = int(
            total_tokens * judge_output.hallucination_token_fraction_estimate
        )
        
        # Build result
        result = EvaluationResult(
            evaluation_id=evaluation_id,
            timestamp=datetime.utcnow(),
            input_data=eval_input,
            judge_output=judge_output,
            total_output_tokens=total_tokens,
            estimated_hallucinated_tokens=estimated_hallucinated_tokens,
            judge_model_used=f"{self.judge.provider}:{self.judge.model_name}",
            evaluation_duration_seconds=judge_result["evaluation_duration"]
        )
        
        return result
    
    async def evaluate_with_target_generation(
        self,
        prompt: str,
        target_provider: str,
        target_model: str,
        prompt_id: Optional[str] = None,
        **generation_kwargs
    ) -> EvaluationResult:
        """
        Generate output from target LLM and evaluate it.
        
        Args:
            prompt: User prompt to send to target LLM
            target_provider: Provider for target LLM
            target_model: Model name for target LLM
            prompt_id: Optional prompt ID (generated if not provided)
            **generation_kwargs: Additional args for generation
        
        Returns:
            Complete evaluation result
        """
        # Initialize target LLM
        target_llm = TargetLLM(provider=target_provider, model_name=target_model)
        
        # Generate output
        generation_result = await target_llm.generate(prompt, **generation_kwargs)
        
        # Create evaluation input
        eval_input = EvaluationInput(
            prompt_id=prompt_id or str(uuid.uuid4()),
            user_prompt=prompt,
            target_output=generation_result["output"]
        )
        
        # Evaluate
        return await self.evaluate(eval_input, target_llm)
