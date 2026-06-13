"""
Ollama provider support for TinyTroupe.

This module enables TinyTroupe to use Ollama as an LLM provider,
supporting local model inference.
"""

import requests
import json
from typing import Dict, Any, Optional


class OllamaProvider:
    """
    Ollama provider for local LLM inference.
    """

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        """
        Initialize Ollama provider.

        Args:
            base_url: Ollama API base URL
            model: Model name to use
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_url = f"{base_url}/api/generate"

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            raise Exception(f"Ollama generation failed: {e}")

    def check_connection(self) -> bool:
        """
        Check if Ollama service is accessible.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False