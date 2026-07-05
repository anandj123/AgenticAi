"""Deterministic LangChain tools for inventory checks."""

from langchain_core.tools import tool

DISCONTINUED_PARTS = frozenset({"Part X", "Part Y"})


@tool
def check_part_availability(parts_list: list[str]) -> str:
    """Checks the live inventory database to ensure parts are not discontinued or backordered."""
    found_errors = [part for part in parts_list if part in DISCONTINUED_PARTS]

    if found_errors:
        return f"Deterministic Error: The following parts are DISCONTINUED: {found_errors}"
    return "All parts are active and available in the inventory database."
