import unittest
import logging
import os
from unittest.mock import patch, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinytroupe")
logger.setLevel(logging.INFO)

try:
    import ollama
    OLLAMA_IMPORTED = True
except ImportError:
    OLLAMA_IMPORTED = False

from tinytroupe.openai_utils import force_api_type, client, register_client
from tinytroupe.ollama_utils import OllamaClient

@unittest.skipIf(not OLLAMA_IMPORTED, "Ollama package not installed")
class TestOllamaIntegration(unittest.TestCase):
    """Tests for the Ollama integration."""

    def setUp(self):
        """Set up for the tests."""
        self.original_client = client()
        # Keep track of the original value
        if "OLLAMA_HOST" in os.environ:
            self.original_host = os.environ["OLLAMA_HOST"] 
        else:
            self.original_host = None

    def tearDown(self):
        """Clean up after the tests."""
        # Restore the original client
        force_api_type(self.original_client.__class__.__name__.replace("Client", "").lower())
        # Restore the original OLLAMA_HOST
        if self.original_host:
            os.environ["OLLAMA_HOST"] = self.original_host
        elif "OLLAMA_HOST" in os.environ:
            del os.environ["OLLAMA_HOST"]

    @patch('ollama.Client')
    def test_ollama_client_initialization(self, mock_ollama_client):
        """Test that the OllamaClient is initialized correctly."""
        # Create a mock for the Ollama client
        mock_client = MagicMock()
        mock_ollama_client.return_value = mock_client

        # Register a new Ollama client
        ollama_client = OllamaClient()
        register_client("ollama_test", ollama_client)
        
        # Force using the test client
        force_api_type("ollama_test")
        
        # Check if the current client is the OllamaClient
        current_client = client()
        self.assertIsInstance(current_client, OllamaClient)
        
        # Verify the client is set up
        mock_ollama_client.assert_called_once()

    @patch('ollama.Client')
    def test_ollama_chat_call(self, mock_ollama_client):
        """Test that chat calls are correctly made to Ollama."""
        # Create a mock for the Ollama client
        mock_client = MagicMock()
        mock_ollama_client.return_value = mock_client
        
        # Mock the chat response
        mock_response = {
            "message": {"content": "This is a test response from Ollama"}
        }
        mock_client.chat.return_value = mock_response
        
        # Register a new Ollama client
        ollama_client = OllamaClient()
        register_client("ollama_test", ollama_client)
        
        # Force using the test client
        force_api_type("ollama_test")
        
        # Make a chat call
        current_client = client()
        messages = [{"role": "user", "content": "Hello, Ollama!"}]
        response = current_client._raw_model_call("long-gemma", {"messages": messages})
        
        # Verify the chat method was called with the correct parameters
        mock_client.chat.assert_called_once()
        args, kwargs = mock_client.chat.call_args
        self.assertEqual(kwargs["model"], "long-gemma")
        self.assertEqual(kwargs["messages"], messages)
        
        # Verify the response is as expected
        self.assertEqual(response, mock_response)

    @patch('ollama.Client')
    def test_embedding_fallback(self, mock_ollama_client):
        """Test that embedding fallback works when Ollama embedding fails."""
        # Create a mock for the Ollama client
        mock_client = MagicMock()
        mock_ollama_client.return_value = mock_client
        
        # Mock the embeddings method to raise an exception
        mock_client.embeddings.side_effect = Exception("Ollama embedding failed")
        
        # Register a new Ollama client with fallback enabled
        ollama_client = OllamaClient()
        ollama_client.embedding_fallback = True
        register_client("ollama_test", ollama_client)
        
        # Force using the test client
        force_api_type("ollama_test")
        
        # Mock the OpenAIClient.get_embedding method to return a dummy embedding
        with patch('tinytroupe.openai_utils.OpenAIClient.get_embedding', return_value=[0.1, 0.2, 0.3]):
            # Get an embedding
            current_client = client()
            embedding = current_client.get_embedding("Test text")
            
            # Verify the embeddings method was called
            mock_client.embeddings.assert_called_once()
            
            # Verify the embedding fallback was used
            self.assertEqual(embedding, [0.1, 0.2, 0.3])

if __name__ == "__main__":
    unittest.main()
