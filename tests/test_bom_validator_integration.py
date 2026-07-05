import os

import pytest

from agentic_ai.agent.bom_validator import TEST_BOMS, create_bom_validator_agent, validate_bom


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_bom_validator_runs_all_cases() -> None:
    agent_executor = create_bom_validator_agent(verbose=False)

    for _label, bom_input in TEST_BOMS:
        output = validate_bom(agent_executor, bom_input)
        assert "VERDICT:" in output
        assert "REASONING:" in output
