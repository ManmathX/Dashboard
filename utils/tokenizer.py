"""
Tokenization utilities.
"""

import tiktoken
from typing import Optional


class Tokenizer:
    """Token counting utilities for different models."""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize tokenizer.
        
        Args:
            model_name: Model name for tokenizer selection
        """
        self.model_name = model_name
        self.encoding = self._get_encoding(model_name)
    
    def _get_encoding(self, model_name: Optional[str]):
        """Get appropriate encoding for model."""
        if not model_name:
            return tiktoken.get_encoding("cl100k_base")
        
        try:
            return tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            return tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to tokenize
        
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def count_tokens_in_segments(self, segments: list) -> int:
        """
        Count total tokens across multiple text segments.
        
        Args:
            segments: List of text segments
        
        Returns:
            Total token count
        """
        return sum(self.count_tokens(seg) for seg in segments)
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Quick token estimation without loading encoding.
        Uses approximation: 1 token ≈ 4 characters.
        
        Args:
            text: Text to estimate
        
        Returns:
            Estimated token count
        """
        return len(text) // 4
