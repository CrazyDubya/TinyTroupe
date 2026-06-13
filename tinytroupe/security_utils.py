"""
Input validation utilities for TinyTroupe with security best practices.
"""

import re
from typing import Any, Dict, List
from .exceptions import ValidationException, SecurityException


def validate_prompt_length(prompt: str, max_length: int = 100000) -> None:
    """
    Validates that prompt length is within acceptable limits.

    Args:
        prompt: The prompt text to validate
        max_length: Maximum allowed length in characters

    Raises:
        ValidationException: If prompt exceeds maximum length
    """
    if len(prompt) > max_length:
        raise ValidationException(
            f"Prompt length ({len(prompt)}) exceeds maximum allowed length ({max_length})"
        )


def sanitize_input(input_str: str, max_length: int = 10000) -> str:
    """
    Sanitizes user input to prevent injection attacks.

    Args:
        input_str: User input string to sanitize
        max_length: Maximum allowed length after sanitization

    Returns:
        Sanitized string

    Raises:
        SecurityException: If input contains potentially malicious patterns
    """
    if not isinstance(input_str, str):
        raise SecurityException("Input must be a string")

    # Check for suspicious patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                  # JavaScript protocol
        r'on\w+\s*=',                   # Event handlers
        r'<iframe[^>]*>',               # Iframes
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, input_str, re.IGNORECASE):
            raise SecurityException(
                f"Input contains potentially malicious pattern: {pattern}"
            )

    # Truncate to max length
    if len(input_str) > max_length:
        input_str = input_str[:max_length]

    return input_str


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> None:
    """
    Validates that JSON data contains required fields.

    Args:
        data: Dictionary to validate
        required_fields: List of required field names

    Raises:
        ValidationException: If required fields are missing
    """
    if not isinstance(data, dict):
        raise ValidationException("Data must be a dictionary")

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValidationException(
            f"Missing required fields: {', '.join(missing_fields)}"
        )


def validate_llm_response(response: str, max_tokens: int = 4096) -> None:
    """
    Validates LLM response length and content.

    Args:
        response: LLM response string
        max_tokens: Maximum allowed response length in tokens

    Raises:
        ValidationException: If response exceeds maximum length
    """
    if not isinstance(response, str):
        raise ValidationException("Response must be a string")

    # Rough token estimation (1 token ≈ 4 characters)
    estimated_tokens = len(response) // 4

    if estimated_tokens > max_tokens:
        raise ValidationException(
            f"Response length ({estimated_tokens} tokens) exceeds maximum ({max_tokens})"
        )