# Ollama Integration

TinyTroupe now supports local language models through [Ollama](https://ollama.com/) integration. This allows you to run persona simulations using local models on your own hardware, providing improved privacy and reduced cost compared to cloud-based models.

## Prerequisites

1. Install Ollama from [ollama.com](https://ollama.com/)
2. Pull the models you want to use:
   ```bash
   ollama pull long-gemma
   ```
3. Install TinyTroupe with Ollama support:
   ```bash
   pip install tinytroupe[ollama]
   ```

## Configuration

To configure TinyTroupe to use Ollama, you can update your `config.ini` file:

```ini
[OpenAI]
# Change to "ollama" to use Ollama as the default provider
API_TYPE=openai  # or "ollama" to use Ollama by default

[Ollama]
# Ollama configuration
MODEL=long-gemma  # Default model to use
HOST=http://localhost:11434  # Ollama server address
EMBEDDING_MODEL=long-gemma  # Model to use for embeddings
EMBEDDING_FALLBACK=True  # Whether to fall back to OpenAI for embeddings if local fails
```

## Usage

You can use Ollama in two ways:

### 1. Set it as the default provider in your config.ini
Set `API_TYPE=ollama` in the `[OpenAI]` section of your config.ini file.

### 2. Force using Ollama for a specific session

```python
import tinytroupe as tt
from tinytroupe.openai_utils import force_api_type

# Force using Ollama for this session
force_api_type("ollama")

# Create and use personas as usual
persona = tt.Persona(
    name="LocalExpert",
    backstory="I am an AI assistant running on a local model.",
    traits=["helpful", "concise"],
    role="assistant"
)
```

## Working with Ollama Models

Ollama supports many models, including:

- long-gemma
- llama3
- mistral
- mixtral
- phi3
- codellama
- and many more

You can specify which model to use in your config.ini or when creating a client:

```python
from tinytroupe.openai_utils import get_client

# Get the current client
client = get_client()

# If it's an Ollama client, you can set the model
if hasattr(client, "ollama_model"):
    client.ollama_model = "llama3"
```

## Embedding Support

Embeddings are supported through Ollama, but not all models provide quality embeddings. By default, if an embedding operation fails with Ollama, TinyTroupe will fall back to using OpenAI's embedding model if `EMBEDDING_FALLBACK=True` is set in your config.

## Best Practices

1. **Model Selection**: Choose an appropriate model for your use case. Larger models provide better reasoning but require more resources.
2. **Resource Considerations**: Local models require significant RAM and GPU resources for optimal performance.
3. **Embedding Fallback**: For critical applications, keep the embedding fallback enabled to ensure consistent performance.

## Limitations

1. **Performance**: Local models may be slower than cloud-based models, especially without GPU acceleration.
2. **Feature Parity**: Some advanced features like function calling might not work the same across all models.
3. **Embedding Quality**: Local embedding quality may vary compared to OpenAI's dedicated embedding models.

## Examples

See the `/examples/using_ollama.ipynb` notebook for a complete example of using TinyTroupe with Ollama.
