"""
Simple demonstration of TinyTroupe with Ollama integration.

This script demonstrates how to use TinyTroupe with local Ollama models.
"""

import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tinytroupe")
logger.setLevel(logging.INFO)

# Import TinyTroupe
import tinytroupe as tt
from tinytroupe.openai_utils import force_api_type, client

def main():
    # Force using Ollama for this session
    force_api_type("ollama")
    
    # Verify we're using the Ollama client
    print(f"Using client: {client().__class__.__name__}")
    
    # Create a persona
    local_persona = tt.Persona(
        name="LocalExpert",
        backstory="I am an AI assistant running on a local model through Ollama. I help answer questions using locally available computing resources.",
        traits=["helpful", "concise", "knowledgeable"],
        role="assistant"
    )
    
    # Create a conversation
    conversation = tt.Conversation()
    
    # Add the persona to the conversation
    conversation.add_persona(local_persona)
    
    # Start the conversation with a question
    print("Sending message to Ollama-powered persona...")
    response = conversation.send_message(
        sender="user",
        content="Hello, can you tell me what are the advantages of using local models like yourself instead of cloud-based LLMs?",
        target="LocalExpert"
    )
    
    # Print the response
    print(f"\nLocalExpert says: {response}\n")
    
    # Continue the conversation
    print("Sending follow-up message...")
    response = conversation.send_message(
        sender="user",
        content="What are the trade-offs between using local models versus cloud-based models?",
        target="LocalExpert"
    )
    
    print(f"\nLocalExpert says: {response}\n")
    
    # Switch back to OpenAI (if keys are available)
    print("Switching back to OpenAI...")
    try:
        force_api_type("openai")
        print(f"Now using client: {client().__class__.__name__}")
    except Exception as e:
        print(f"Could not switch to OpenAI: {e}")
        print("Note: This is expected if you don't have OpenAI API keys configured.")

if __name__ == "__main__":
    main()
