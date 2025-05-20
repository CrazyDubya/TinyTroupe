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
from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.environment.tiny_world import TinyWorld
from tinytroupe.openai_utils import configure, client

def main():
    # Configure to use Ollama for this session
    configure(api_type="ollama")
    
    # Verify we're using the Ollama client
    print(f"Using client: {client().__class__.__name__}")
    
    # Create a world for our personas
    world = TinyWorld()
    
    # Create a persona
    local_persona = TinyPerson(
        name="LocalExpert",
        backstory="I am an AI assistant running on a local model through Ollama. I help answer questions using locally available computing resources.",
        traits=["helpful", "concise", "knowledgeable"],
        role="assistant",
        world=world
    )
    
    # Add the persona to the world
    world.add_agent(local_persona)
    
    # Start the conversation with a question
    print("Sending message to Ollama-powered persona...")
    response = world.broadcast(
        "Hello, can you tell me what are the advantages of using local models like yourself instead of cloud-based LLMs?",
        source=None
    )
    
    # Print the response
    print(f"\nLocalExpert's response will appear in the logs\n")
    
    # Continue the conversation
    print("Sending follow-up message...")
    response = world.broadcast(
        "What are the trade-offs between using local models versus cloud-based models?",
        source=None
    )
    
    print(f"\nLocalExpert's response will appear in the logs\n")
    
    # Switch back to OpenAI (if keys are available)
    print("Switching back to OpenAI...")
    try:
        configure(api_type="openai")
        print(f"Now using client: {client().__class__.__name__}")
    except Exception as e:
        print(f"Could not switch to OpenAI: {e}")
        print("Note: This is expected if you don't have OpenAI API keys configured.")

if __name__ == "__main__":
    main()
