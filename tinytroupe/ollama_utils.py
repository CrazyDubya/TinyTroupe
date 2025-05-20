"""
Utility module for interfacing with Ollama for local model inference.
"""
import os
import logging
import json
import time
import numpy as np
from typing import Optional, Dict, List, Any, Union

import ollama

from tinytroupe import utils
from tinytroupe.openai_utils import OpenAIClient, default, NonTerminalError

logger = logging.getLogger("tinytroupe")

class OllamaClient(OpenAIClient):
    """
    A client for interfacing with Ollama for local model inference.
    
    This client implements the same interface as OpenAIClient but uses the Ollama API
    to make requests to local models.
    """

    def __init__(self, cache_api_calls=default["cache_api_calls"], cache_file_name=default["cache_file_name"]) -> None:
        logger.debug("Initializing OllamaClient")
        super().__init__(cache_api_calls, cache_file_name)
        
        # Get config for Ollama
        config = utils.read_config_file()
        self.ollama_host = config["Ollama"].get("HOST", "http://localhost:11434")
        self.ollama_model = config["Ollama"].get("MODEL", "long-gemma")
        self.embedding_fallback = config["Ollama"].getboolean("EMBEDDING_FALLBACK", True)
        self.embedding_model = config["Ollama"].get("EMBEDDING_MODEL", "long-gemma")
        
        # Allow overriding with environment variables
        if os.getenv("OLLAMA_HOST"):
            self.ollama_host = os.getenv("OLLAMA_HOST")
            
        logger.debug(f"Ollama config - Host: {self.ollama_host}, Model: {self.ollama_model}, Embedding fallback: {self.embedding_fallback}")
        
    def _setup_from_config(self):
        """
        Sets up the Ollama client configurations.
        """
        self.client = ollama.Client(host=self.ollama_host)
        logger.debug(f"Ollama client initialized with host: {self.ollama_host}")

    def _raw_model_call(self, model, chat_api_params):
        """
        Calls the Ollama API with the given parameters.
        """
        # Extract and convert OpenAI API parameters to Ollama API parameters
        messages = chat_api_params.get("messages", [])
        
        # Ollama doesn't support "stream" parameter in the same way as OpenAI
        if "stream" in chat_api_params:
            del chat_api_params["stream"]
            
        # Prepare Ollama-specific parameters
        ollama_params = {
            "model": model,
            "messages": messages,
            "options": {
                "temperature": chat_api_params.get("temperature", 0.7),
                "top_p": chat_api_params.get("top_p", 0.9),
                "frequency_penalty": chat_api_params.get("frequency_penalty", 0.0),
                "presence_penalty": chat_api_params.get("presence_penalty", 0.0),
                "stop": chat_api_params.get("stop", []),
            }
        }
        
        # Add num_predict (equivalent to max_tokens) if specified
        if "max_tokens" in chat_api_params:
            ollama_params["options"]["num_predict"] = chat_api_params["max_tokens"]
        
        try:
            # Make the actual call to Ollama
            response = self.client.chat(**ollama_params)
            return response
        except Exception as e:
            logger.error(f"Error calling Ollama API: {e}")
            raise NonTerminalError(f"Error calling Ollama API: {e}")

    def _raw_model_response_extractor(self, response):
        """
        Extracts the response from the Ollama API response.
        """
        # Ollama response format is already similar to OpenAI's but with different structure
        return {
            "role": "assistant",
            "content": response["message"]["content"]
        }

    def get_embedding(self, text, model=None):
        """
        Gets the embedding of the given text using the specified model.
        
        Args:
        text (str): The text to embed.
        model (str): The name of the model to use for embedding the text.
        
        Returns:
        The embedding of the text.
        """
        # If no model is specified, use the default embedding model from config
        if model is None:
            model = default["embedding_model"]
            
        try:
            # Use Ollama's embedding endpoint
            response = self.client.embeddings(model=model, prompt=text)
            return response.get("embedding", [])
        except Exception as e:
            logger.error(f"Error getting embedding from Ollama: {e}")
            # Fallback to a simpler mechanism or return empty embedding
            logger.warning("Falling back to OpenAI for embeddings if available")
            try:
                # Try using OpenAI's embedding if available
                openai_client = OpenAIClient()
                return openai_client.get_embedding(text, model=default["embedding_model"])
            except Exception as eb:
                logger.error(f"Fallback to OpenAI embedding also failed: {eb}")
                return []

    def _raw_embedding_model_call(self, text, model):
        """
        Calls the Ollama API to get the embedding of the given text.
        """
        return self.client.embeddings(model=model, prompt=text)
    
    def _raw_embedding_model_response_extractor(self, response):
        """
        Extracts the embedding from the Ollama API response.
        """
        return response.get("embedding", [])

    def _count_tokens(self, messages: list, model: str):
        """
        Approximate token counting for Ollama models.
        
        Note: This is a simplified approach as Ollama doesn't provide a direct token counting API.
        """
        # Convert messages to a single string
        combined_text = ""
        for message in messages:
            combined_text += message.get("content", "") + " "
        
        # Approximate token count: ~4 characters per token for English text
        approx_tokens = len(combined_text) // 4
        
        logger.debug(f"Approximate token count for Ollama: {approx_tokens}")
        return approx_tokens
