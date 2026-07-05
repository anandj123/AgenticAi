"""BOM validation agent combining LangChain orchestration and LlamaIndex retrieval."""

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from agentic_ai.config.settings import Settings, get_settings
from agentic_ai.indexing.knowledge_base import build_rule_query_engine
from agentic_ai.tools.langchain.inventory import check_part_availability
from agentic_ai.tools.llamaindex.rules import create_query_engineering_rules_tool

SYSTEM_PROMPT = """You are an expert Manufacturing and Configuration Verification Agent.
Your job is to strictly validate a Bill of Materials (BOM) against provided rules and data.

Follow this exact validation pipeline:
1. Extract the unique part numbers from the user's input BOM.
2. Use the `check_part_availability` tool to verify the parts exist and are active. If this fails, the BOM is INVALID.
3. Use the `query_engineering_rules` tool to check for engineering incompatibilities or missing mandatory pairs (e.g., matching chassis with shock absorbers).
4. Evaluate if the BOM matches any known valid patterns or violates any explicit negative constraints.

Output your final verdict strictly formatted as:
VERDICT: [VALID / INVALID]
REASONING: [Clear, bulleted explanation of your findings, citing specific rules or database blocks]"""

TEST_BOMS: list[tuple[str, str]] = [
    (
        "BOM 1 — engineering rule violation (Part A + Part C)",
        "Please validate this BOM configuration: [Part A, Part B, Part C]",
    ),
    (
        "BOM 2 — discontinued inventory part (Part X)",
        "Please validate this BOM configuration: [Part A, Part B, Part X]",
    ),
    (
        "BOM 3 — valid industrial configuration",
        "Please validate this BOM configuration: [Part F, Part C, Part D, Part E]",
    ),
]


def create_bom_validator_agent(
    settings: Settings | None = None,
    *,
    verbose: bool = True,
) -> AgentExecutor:
    settings = settings or get_settings()
    rule_query_engine = build_rule_query_engine()
    query_engineering_rules = create_query_engineering_rules_tool(rule_query_engine)
    tools = [check_part_availability, query_engineering_rules]

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key or None,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=verbose)


def validate_bom(agent_executor: AgentExecutor, bom_input: str) -> str:
    response = agent_executor.invoke({"input": bom_input})
    return response["output"]
