import tinytroupe.openai_utils as openai_utils


def test_ollama_client_selection():
    openai_utils.force_api_type("ollama")
    try:
        client = openai_utils.client()
        assert client.__class__.__name__ == "OllamaClient"
    finally:
        openai_utils.force_api_type("openai")

