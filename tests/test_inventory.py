from agentic_ai.tools.langchain.inventory import check_part_availability


def test_check_part_availability_active_parts() -> None:
    result = check_part_availability.invoke({"parts_list": ["Part A", "Part B"]})
    assert "active and available" in result


def test_check_part_availability_discontinued_part() -> None:
    result = check_part_availability.invoke({"parts_list": ["Part A", "Part X"]})
    assert "DISCONTINUED" in result
    assert "Part X" in result
