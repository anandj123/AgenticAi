"""Shared LlamaIndex LLM and embedding configuration."""

from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from agentic_ai.config.settings import get_settings

_configured = False


def configure_llamaindex() -> None:
    global _configured
    if _configured:
        return

    settings = get_settings()
    Settings.llm = OpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key or None,
    )
    Settings.embed_model = OpenAIEmbedding(api_key=settings.openai_api_key or None)
    _configured = True
