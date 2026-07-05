"""LangChain tool wrapper around a LlamaIndex query engine."""

from langchain_core.tools import StructuredTool
from llama_index.core.query_engine import BaseQueryEngine


def create_query_engineering_rules_tool(query_engine: BaseQueryEngine) -> StructuredTool:
    """Expose the LlamaIndex rule query engine as a LangChain tool."""

    def query_engineering_rules(query: str) -> str:
        """Queries the semantic knowledge base for compatibility rules, regulations, and historical BOM examples."""
        response = query_engine.query(query)
        return str(response)

    return StructuredTool.from_function(
        func=query_engineering_rules,
        name="query_engineering_rules",
        description=(
            "Queries the semantic knowledge base for compatibility rules, regulations, "
            "and historical BOM examples."
        ),
    )
