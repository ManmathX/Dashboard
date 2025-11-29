"""
Unit tests for tokenizer utilities.
"""

import pytest
from utils.tokenizer import Tokenizer


def test_tokenizer_initialization():
    """Test tokenizer initialization."""
    tokenizer = Tokenizer()
    assert tokenizer.encoding is not None


def test_count_tokens():
    """Test token counting."""
    tokenizer = Tokenizer()
    
    text = "Hello, world!"
    count = tokenizer.count_tokens(text)
    
    assert count > 0
    assert isinstance(count, int)


def test_count_tokens_empty():
    """Test token counting with empty string."""
    tokenizer = Tokenizer()
    
    count = tokenizer.count_tokens("")
    assert count == 0


def test_count_tokens_in_segments():
    """Test counting tokens across segments."""
    tokenizer = Tokenizer()
    
    segments = [
        "This is the first segment.",
        "This is the second segment.",
        "And this is the third."
    ]
    
    total = tokenizer.count_tokens_in_segments(segments)
    
    # Should be sum of individual counts
    expected = sum(tokenizer.count_tokens(seg) for seg in segments)
    assert total == expected


def test_estimate_tokens():
    """Test token estimation."""
    text = "This is a test sentence with some words."
    
    estimate = Tokenizer.estimate_tokens(text)
    
    # Rough check (1 token ≈ 4 characters)
    assert estimate > 0
    assert estimate < len(text)  # Should be less than character count


def test_model_specific_tokenizer():
    """Test tokenizer with specific model."""
    tokenizer_gpt4 = Tokenizer("gpt-4")
    tokenizer_gpt35 = Tokenizer("gpt-3.5-turbo")
    
    text = "Hello, how are you?"
    
    count_gpt4 = tokenizer_gpt4.count_tokens(text)
    count_gpt35 = tokenizer_gpt35.count_tokens(text)
    
    # Both should give reasonable counts
    assert count_gpt4 > 0
    assert count_gpt35 > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
