"""
A simple demonstration of TinyTroupe with Ollama integration.

This script is a minimal example showing how to use TinyTroupe with local Ollama models.
"""

import logging
import os

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
    print(f"Using client: {client().__class__.__name__}")
    
    # Create a world for our personas
    world = TinyWorld()
    
    # Create a persona
    local_persona = TinyPerson(
        name="LocalExpert",
        backstory="I am an AI assistant running on a local model through Ollama.",
        traits=["helpful", "concise", "knowledgeable"],
        role="assistant",
        world=world
    )
    
    # Add the persona to the world
    world.add_agent(local_persona)
    
    # Start the conversation
    print("Sending message to Ollama-powered persona...")
    world.broadcast("What are the advantages of local LLMs over cloud-based ones?")
    
    # The response will be visible in the logs
    print("\nCheck the logs for LocalExpert's response.\n")
    
    print("Switching back to OpenAI...")
    configure(api_type="openai")
    print(f"Now using client: {client().__class__.__name__}")

if __name__ == "__main__":
    main()
