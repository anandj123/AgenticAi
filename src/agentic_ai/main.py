"""CLI entry point for BOM validation."""

import argparse
import sys

from langchain_core.messages import AIMessage, HumanMessage

from agentic_ai.agent.bom_validator import TEST_BOMS, create_bom_validator_agent, validate_bom
from agentic_ai.config.settings import get_settings


def run_test_cases(verbose: bool = True) -> None:
    settings = get_settings()
    agent_executor = create_bom_validator_agent(settings, verbose=verbose)

    for label, bom_input in TEST_BOMS:
        print(f"\n--- Testing {label} ---")
        output = validate_bom(agent_executor, bom_input)
        print("\nFinal Result:")
        print(output)


def run_interactive(verbose: bool = True) -> None:
    settings = get_settings()
    agent_executor = create_bom_validator_agent(settings, verbose=verbose)
    chat_history: list[HumanMessage | AIMessage] = []

    print("BOM Validation Agent (interactive)")
    print("Paste a BOM to validate, e.g.: Please validate this BOM: [Part A, Part B, Part E]")
    print("Commands: quit, exit, or Ctrl+C to stop\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        output = response["output"]
        chat_history.extend([HumanMessage(content=user_input), AIMessage(content=output)])

        print(f"\nAgent:\n{output}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the BOM validation agent.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the 3 built-in test BOMs (default without --interactive).",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Chat with the agent in a REPL loop.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide LangChain tool-call logs.",
    )
    args = parser.parse_args()

    if not get_settings().openai_api_key:
        print("Error: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    verbose = not args.quiet

    if args.interactive:
        run_interactive(verbose=verbose)
    else:
        run_test_cases(verbose=verbose)


if __name__ == "__main__":
    main()
