"""
Ollama provider support for TinyTroupe.

This module enables TinyTroupe to use Ollama as an LLM provider,
supporting local model inference.
"""

import requests
import json
import logging
import socket
from typing import Dict, Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


ALLOWED_PARAMS = {'temperature', 'top_p', 'num_ctx', 'top_k', 'num_predict'}


def _validate_base_url(base_url: str) -> str:
    """
    Validate that base_url is localhost only to prevent SSRF attacks.
    Uses IP address resolution to prevent hostname bypass attacks.

    Args:
        base_url: URL to validate

    Returns:
        Validated URL

    Raises:
        ValueError: If URL is not localhost or resolves to non-loopback address
    """
    # Check for userinfo bypass before parsing
    if '@' in base_url:
        raise ValueError("Userinfo not allowed in URL")

    parsed = urlparse(base_url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError("Only http/https URLs allowed")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Invalid hostname")

    # Resolve hostname to IP address and check if loopback
    try:
        # Get address info for the hostname
        addr_info = socket.getaddrinfo(hostname, None)
        for addr in addr_info:
            ip = addr[4][0]  # Get IP address
            # Check if IPv4 loopback (127.x.x.x)
            if ip.startswith('127.'):
                continue  # Allow loopback
            # Check if IPv6 loopback (::1)
            elif ip == '::1':
                continue  # Allow loopback
            else:
                raise ValueError(f"Only localhost URLs allowed for Ollama provider (resolved to {ip})")
    except socket.gaierror:
        raise ValueError("Unable to resolve hostname")

    return base_url
    return base_url


class OllamaProvider:
    """
    Ollama provider for local LLM inference.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL (must be localhost)
            model: Model name to use
        """
        validated_url = _validate_base_url(base_url)
        self.base_url = validated_url.rstrip("/")
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Generated text response
        """
        # Validate and filter parameters to prevent injection
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        payload.update({k: v for k, v in kwargs.items() if k in ALLOWED_PARAMS})

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.warning("Ollama generation failed")
            raise Exception("Ollama generation failed")

    def check_connection(self) -> bool:
        """
        Check if Ollama service is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except (requests.exceptions.RequestException, Exception):
            return False