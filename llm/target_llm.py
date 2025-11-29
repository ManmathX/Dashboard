"""
Target LLM interface for testing models.
Supports multiple LLM providers (OpenAI, Anthropic).
"""

import time
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from groq import AsyncGroq
import google.generativeai as genai
import tiktoken

from config import settings


class TargetLLM:
    """Interface for interacting with target LLMs under evaluation."""
    
    def __init__(self, provider: str = None, model_name: str = None):
        """
        Initialize target LLM client.
        
        Args:
            provider: LLM provider (openai, anthropic, groq, gemini, perplexity). Defaults to config.
            model_name: Model name. Defaults to config.
        """
        self.provider = provider or settings.default_target_provider
        self.model_name = model_name or settings.default_target_model
        
        # Initialize clients
        if self.provider == "openai":
            api_key = settings.get_api_key("openai")
            if not api_key:
                raise ValueError("OpenAI API key not configured")
            self.client = AsyncOpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            api_key = settings.get_api_key("anthropic")
            if not api_key:
                raise ValueError("Anthropic API key not configured")
            self.client = AsyncAnthropic(api_key=api_key)
        elif self.provider == "groq":
            api_key = settings.get_api_key("groq")
            if not api_key:
                raise ValueError("Groq API key not configured")
            self.client = AsyncGroq(api_key=api_key)
        elif self.provider == "gemini":
            api_key = settings.get_api_key("gemini")
            if not api_key:
                raise ValueError("Gemini API key not configured")
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model_name)
        elif self.provider == "perplexity":
            # Perplexity uses OpenAI-compatible API
            api_key = settings.get_api_key("perplexity")
            if not api_key:
                raise ValueError("Perplexity API key not configured")
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.perplexity.ai"
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate output from target LLM.
        
        Args:
            prompt: User prompt to send to the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Dict containing:
                - output: Generated text
                - tokens: Token count
                - latency: Generation time in seconds
                - model: Model name used
                - provider: Provider name
        """
        start_time = time.time()
        
        if self.provider == "openai":
            response = await self._generate_openai(prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "anthropic":
            response = await self._generate_anthropic(prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "groq":
            response = await self._generate_groq(prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "gemini":
            response = await self._generate_gemini(prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "perplexity":
            # Perplexity uses OpenAI-compatible API
            response = await self._generate_openai(prompt, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        latency = time.time() - start_time
        
        return {
            "output": response["text"],
            "tokens": response["tokens"],
            "latency": latency,
            "model": self.model_name,
            "provider": self.provider
        }
    
    async def _generate_openai(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using OpenAI API."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens
        }
    
    async def _generate_anthropic(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Anthropic API."""
        response = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.content[0].text,
            "tokens": response.usage.output_tokens
        }
    
    async def _generate_groq(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Groq API."""
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return {
            "text": response.choices[0].message.content,
            "tokens": response.usage.completion_tokens
        }
    
    async def _generate_gemini(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate using Google Gemini API."""
        import asyncio
        
        # Gemini SDK is not fully async, so we run in executor
        loop = asyncio.get_event_loop()
        
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        
        response = await loop.run_in_executor(
            None,
            lambda: self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
        )
        
        # Extract text and estimate tokens
        text = response.text
        tokens = len(text.split())  # Rough estimate
        
        return {
            "text": text,
            "tokens": tokens
        }
    
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using appropriate tokenizer.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Number of tokens
        """
        if self.provider == "openai":
            try:
                encoding = tiktoken.encoding_for_model(self.model_name)
            except KeyError:
                # Fallback to cl100k_base for unknown models
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        elif self.provider in ["anthropic", "groq"]:
            # Anthropic and Groq use similar tokenization to GPT
            # Using cl100k_base as approximation
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
