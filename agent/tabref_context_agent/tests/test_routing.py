import pytest

from tabref_agent.nodes import decide_review_action


@pytest.mark.parametrize(
    ("mode", "ready", "expected"),
    [
        ("prepare", False, "stop"),
        ("prepare", True, "stop"),
        ("codex", False, "codex"),
        ("codex", True, "materialize"),
        ("existing", False, "stop"),
        ("existing", True, "materialize"),
        ("manual", False, "manual"),
    ],
)
def test_review_routing(mode, ready, expected):
    assert decide_review_action(mode, ready) == expected


def test_force_codex_review_bypasses_existing_decisions():
    assert decide_review_action("codex", True, force_codex_review=True) == "codex"
