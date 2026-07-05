"""LlamaIndex knowledge base for engineering rules and BOM examples."""

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.query_engine import BaseQueryEngine

from agentic_ai.indexing.llamaindex_config import configure_llamaindex

KNOWLEDGE_BASE_DOCUMENTS = [
    Document(
        text=(
            "Rule 101: Part A (Main Board v1) is strictly incompatible with "
            "Part C (Legacy Power Supply). It will cause voltage overloads."
        )
    ),
    Document(
        text=(
            "Rule 102: Any configuration using Part D (Heavy Duty Chassis) must include "
            "Part E (Shock Absorber Pack) to meet safety standards."
        )
    ),
    Document(
        text=(
            "Valid Example 1: A BOM containing [Part A, Part B, Part E] is a validated "
            "and working drone configuration."
        )
    ),
    Document(
        text=(
            "Valid Example 2: A BOM containing [Part F, Part C, Part D, Part E] is a "
            "validated industrial configuration."
        )
    ),
]


def build_rule_query_engine(similarity_top_k: int = 2) -> BaseQueryEngine:
    configure_llamaindex()
    index = VectorStoreIndex.from_documents(KNOWLEDGE_BASE_DOCUMENTS)
    return index.as_query_engine(similarity_top_k=similarity_top_k)
