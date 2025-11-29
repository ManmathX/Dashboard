"""
Judge LLM evaluation engine.
Uses a powerful LLM to evaluate target outputs with strict scoring.
"""

import json
import logging
import time
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from groq import AsyncGroq

from config import settings
from models.schemas import EvaluationInput, JudgeOutput


class JudgeLLM:
    """Judge LLM for evaluating target model outputs."""
    
    def __init__(self):
        """Initialize Judge LLM client based on configuration."""
        self.provider = settings.judge_model_provider
        self.model_name = settings.judge_model_name
        
        # Load system prompt
        with open("prompts/judge_system_prompt.txt", "r") as f:
            self.system_prompt = f.read()
        
        # Load user template
        with open("prompts/judge_user_template.txt", "r") as f:
            self.user_template = f.read()
        
        # Initialize client based on provider
        if self.provider == "openai":
            api_key = settings.get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not configured for Judge LLM")
            self.client = AsyncOpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = settings.get_api_key("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not configured for Judge LLM")
            self.client = AsyncAnthropic(api_key=api_key)
        elif self.provider == "groq":
            api_key = settings.get_api_key("groq")
            if not api_key:
                raise ValueError("Groq API key not configured for Judge LLM")
            self.client = AsyncGroq(api_key=api_key)
        elif self.provider == "perplexity":
            api_key = settings.get_api_key("perplexity")
            if not api_key:
                raise ValueError("Perplexity API key not configured for Judge LLM")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            raise ValueError(f"Unsupported judge provider: {self.provider}")
    
    async def evaluate(self, eval_input: EvaluationInput) -> Dict[str, Any]:
        """
        Evaluate target LLM output using Judge LLM.
        
        Args:
            eval_input: Evaluation input containing prompt, target output, etc.
        
        Returns:
            Dict containing:
                - judge_output: Parsed JudgeOutput model
                - raw_response: Raw JSON string from judge
                - evaluation_duration: Time taken for evaluation
        """
        start_time = time.time()
        
        # Format user message
        user_message = self._format_user_message(eval_input)
        
        # Call judge LLM
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    raw_response = await self._evaluate_openai(user_message)
                elif self.provider == "anthropic":
                    raw_response = await self._evaluate_anthropic(user_message)
                elif self.provider in ["groq", "perplexity"]:
                    # Groq and Perplexity use OpenAI-compatible API
                    raw_response = await self._evaluate_openai(user_message)
                else:
                    raise ValueError(f"Unsupported judge provider: {self.provider}")
                
                # Parse and validate JSON response
                judge_output = self._parse_judge_response(raw_response)
                
                duration = time.time() - start_time
                
                return {
                    "judge_output": judge_output,
                    "raw_response": raw_response,
                    "evaluation_duration": duration
                }
            
            except (json.JSONDecodeError, ValueError) as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Judge LLM failed to return valid JSON after {max_retries} attempts: {e}")
                # Retry on JSON parsing errors
                continue
        
        raise ValueError("Judge LLM evaluation failed")
    
    def _format_user_message(self, eval_input: EvaluationInput) -> str:
        """Format the user message for judge LLM."""
        # Format other model outputs
        other_outputs_str = "None provided"
        if eval_input.other_model_outputs:
            other_outputs_str = "\n".join([
                f"Model: {mo.model_name}\nOutput: {mo.output}\n"
                for mo in eval_input.other_model_outputs
            ])
        
        # Format ground truth
        ground_truth_str = "None provided"
        if eval_input.ground_truth:
            ground_truth_str = f"Type: {eval_input.ground_truth.type}\n"
            ground_truth_str += f"Content: {eval_input.ground_truth.content}\n"
            if eval_input.ground_truth.sources:
                ground_truth_str += f"Sources: {', '.join(eval_input.ground_truth.sources)}"
        
        # Fill template
        return self.user_template.format(
            user_prompt=eval_input.user_prompt,
            target_output=eval_input.target_output,
            other_model_outputs=other_outputs_str,
            ground_truth=ground_truth_str,
            task_type=eval_input.metadata.task_type,
            language=eval_input.metadata.language,
            eval_purpose=eval_input.metadata.eval_purpose
        )
    
    async def _evaluate_openai(self, user_message: str) -> str:
        """Evaluate using OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=settings.judge_temperature,
            max_tokens=settings.judge_max_tokens,
            response_format={"type": "json_object"}  # Force JSON mode
        )
        
        return response.choices[0].message.content
    
    async def _evaluate_anthropic(self, user_message: str) -> str:
        """Evaluate using Anthropic API."""
        response = await self.client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
            temperature=settings.judge_temperature,
            max_tokens=settings.judge_max_tokens
        )
        
        return response.content[0].text
    
    def _parse_judge_response(self, raw_response: str) -> JudgeOutput:
        """
        Parse and validate judge response.
        
        Args:
            raw_response: Raw JSON string from judge
        
        Returns:
            Validated JudgeOutput model
        
        Raises:
            ValueError: If response is invalid
        """
        # Parse JSON
        try:
            response_dict = json.loads(raw_response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from judge: {e}")
        
        # Validate using Pydantic
        try:
            judge_output = JudgeOutput(**response_dict)
        except Exception as e:
            raise ValueError(f"Judge response doesn't match schema: {e}")
        
        return judge_output
