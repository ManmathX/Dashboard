"""
Utility functions for validation and sanitization.
"""

import re
from typing import Any, Dict


class Validators:
    """Validation utilities."""
    
    @staticmethod
    def sanitize_api_key(text: str) -> str:
        """
        Remove API keys from text to prevent leakage.
        
        Args:
            text: Text that might contain API keys
        
        Returns:
            Sanitized text
        """
        # Pattern for common API key formats
        patterns = [
            r'sk-[a-zA-Z0-9]{20,}',  # OpenAI keys
            r'sk-ant-[a-zA-Z0-9-]{20,}',  # Anthropic keys
            r'AIza[a-zA-Z0-9_-]{35}',  # Google API keys
            r'[a-zA-Z0-9]{32,}',  # Generic long strings
        ]
        
        sanitized = text
        for pattern in patterns:
            sanitized = re.sub(pattern, '[REDACTED_API_KEY]', sanitized)
        
        return sanitized
    
    @staticmethod
    def validate_probability(value: float, field_name: str = "probability") -> None:
        """
        Validate that a probability is in valid range.
        
        Args:
            value: Probability value
            field_name: Name of field for error message
        
        Raises:
            ValueError: If probability is invalid
        """
        if not 0 <= value <= 100:
            raise ValueError(f"{field_name} must be between 0 and 100, got {value}")
    
    @staticmethod
    def validate_fraction(value: float, field_name: str = "fraction") -> None:
        """
        Validate that a fraction is in valid range.
        
        Args:
            value: Fraction value
            field_name: Name of field for error message
        
        Raises:
            ValueError: If fraction is invalid
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field_name} must be between 0.0 and 1.0, got {value}")
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_fields: list) -> None:
        """
        Validate that JSON has required fields.
        
        Args:
            data: JSON data dict
            required_fields: List of required field names
        
        Raises:
            ValueError: If required fields are missing
        """
        missing = [field for field in required_fields if field not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
