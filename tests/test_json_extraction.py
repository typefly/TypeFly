import pytest

from typefly.plan_execution import extract_plan_json, PlanValidationError


def test_fenced_json():
    raw = '```json\n{"thoughts":"t","plan":"log(\'hi\')"}\n```'
    assert extract_plan_json(raw)["plan"] == "log('hi')"


def test_bare_json():
    assert extract_plan_json('{"plan":"log(\'hi\')"}')["plan"] == "log('hi')"


def test_uppercase_fence():
    raw = '```JSON\n{"plan":"log(\'hi\')"}\n```'
    assert extract_plan_json(raw)["plan"] == "log('hi')"


def test_prose_around_bare_json():
    raw = 'Sure! Here you go:\n{"plan":"log(\'hi\')"}\nHope that helps.'
    assert extract_plan_json(raw)["plan"] == "log('hi')"


@pytest.mark.parametrize("raw", [
    "",
    "   ",
    "not json at all",
    '{"thoughts":"t"}',      # missing plan
    '{"plan": null}',        # plan not a string
    '{"plan": ""}',          # empty plan
    '{"plan": 123}',         # plan wrong type
    '["plan"]',              # not an object
    '{"plan":',              # malformed
    None,                    # not a string
])
def test_invalid_inputs_raise(raw):
    with pytest.raises(PlanValidationError):
        extract_plan_json(raw)
