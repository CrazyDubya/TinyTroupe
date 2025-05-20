# Setting up Ollama with TinyTroupe

This guide will help you set up Ollama to work with TinyTroupe for local LLM inference.

## 1. Install Ollama

First, you need to install Ollama on your system. Follow the instructions at [ollama.com](https://ollama.com/) for your operating system.

### macOS
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows
Download the installer from [ollama.com/download/windows](https://ollama.com/download/windows)

## 2. Start Ollama Server

After installation, start the Ollama server:

```bash
ollama serve
```

This will start the Ollama API server on `http://localhost:11434`.

## 3. Pull the Models

Pull the models you want to use with TinyTroupe. The default model configured for TinyTroupe is "long-gemma", but you can use any model supported by Ollama.

```bash
# Pull the long-gemma model (default for TinyTroupe)
ollama pull long-gemma

# Alternatively, you can pull other models
ollama pull llama3
ollama pull mistral
ollama pull phi3
```

## 4. Configure TinyTroupe

### Update config.ini

Create or modify your `config.ini` file to use Ollama:

```ini
[OpenAI]
# Change this to "ollama" to use Ollama as the default provider
API_TYPE=openai

[Ollama]
# Ollama configuration
MODEL=long-gemma
HOST=http://localhost:11434
EMBEDDING_MODEL=long-gemma
EMBEDDING_FALLBACK=True
```

### Using Environment Variables

Alternatively, you can use environment variables:

```bash
export OLLAMA_HOST=http://localhost:11434
```

## 5. Test Your Setup

Run the included Ollama demo script to verify your setup:

```bash
python examples/ollama_demo.py
```

Or try the Jupyter notebook example:

```bash
jupyter notebook examples/using_ollama.ipynb
```

## Troubleshooting

### Connection Errors

If you see connection errors, make sure:
- Ollama is running with `ollama serve`
- The HOST in config.ini matches where Ollama is running
- Firewall settings allow connections to the Ollama port

### Model Not Found

If you get model not found errors:
- Verify you've pulled the model with `ollama list`
- Make sure the model name in config.ini matches exactly what's in Ollama

### Slow Responses

Local models can be slower than cloud-based ones, especially without GPU acceleration. Consider:
- Using smaller models for faster responses
- Setting up GPU acceleration for Ollama
- Adjusting temperature and other parameters for faster generation

## Using GPU Acceleration

For better performance, configure Ollama to use your GPU:

### NVIDIA GPUs
Ollama should automatically use NVIDIA GPUs if CUDA is installed.

### AMD GPUs (Linux)
Set the environment variable:
```bash
export OLLAMA_HOST=unix:///tmp/ollama.sock
```

## Advanced Configuration

See the [Ollama documentation](https://github.com/ollama/ollama) for advanced configuration options, including:
- Custom model quantization
- Model fine-tuning
- Running Ollama in Docker
