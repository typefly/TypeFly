import pytest

from typefly.plan_execution import validate_plan, PlanValidationError, MAX_LOOP_ITERS

# The full common skillset names (lowercase), as registered by RobotWrapper.
SKILLS = {
    "move_forward", "move_backward", "move_left", "move_right",
    "rotate_left", "rotate_right",
    "is_visible", "object_x", "object_y", "object_width", "object_height", "object_dist",
    "take_picture", "log", "delay", "re_plan", "probe",
    "scan", "scan_description", "orienting", "goto",
}

# Every example plan shipped in assets/example_plans.txt must validate.
EXAMPLE_PLANS = [
    "if scan('bottle'):\n    goto('bottle')\n    height = object_height('bottle')\n    log(height)\n    take_picture()\nelse:\n    log('No bottle in the view')",
    "log('I can already see an apple near the tv.')\ngoto('apple')",
    "rotate_right(180)\nif is_visible('apple'):\n    log('I can see an apple behind me.')\nelse:\n    log('I cannot see an apple behind me.')",
    "obj = scan_description('Any edible object here?')\nif obj:\n    goto(obj)\nelse:\n    log('No edible object in the view')",
    "for _ in range(12):\n    if is_visible('person'):\n        break\n    rotate_right(30)\nif not is_visible('person'):\n    log('No person in the view')\nelse:\n    log('I can see a person in the view')",
    "goto('apple[0.52]')",
    "obj = scan_description('Any chair with a laptop on it?')\nif obj:\n    goto(obj)\nelse:\n    log('No chair with a laptop on it in the view')",
    "move_right(0.5)\ngoto('chair')",
    "log('1231239 + 1231239 = 2462478')",  # math answered by the model, logged as a literal
]


@pytest.mark.parametrize("plan", EXAMPLE_PLANS)
def test_example_plans_validate(plan):
    assert validate_plan(plan, SKILLS) is not None


@pytest.mark.parametrize("plan", [
    "goto(scan_description('x'))",                  # nested skill calls
    "log(f'h {object_height(\"bottle\")}')",        # f-string with skill call
    "x = 1\ny = 2\nlog(x)\nlog(y)",                  # several single-name assigns
    "for _ in range(%d):\n    rotate_right(10)" % MAX_LOOP_ITERS,  # exactly the cap
    "delay(-1)",                                     # negative arg is policy's problem, not the validator's
    "for _ in range(50):\n    for _ in range(50):\n        rotate_right(1)",  # nested product 2500 <= cap
])
def test_additional_valid_plans(plan):
    assert validate_plan(plan, SKILLS) is not None


@pytest.mark.parametrize("plan", [
    "import os",
    "from os import system",
    "__import__('os').system('id')",
    "open('/etc/passwd').read()",
    "log(open('/etc/passwd').read())",
    "().__class__.__bases__[0].__subclasses__()",
    "log((1).__class__)",
    "eval('1+1')",
    "exec('x=1')",
    "getattr(log, '__self__')",
    "jump()",                                        # unknown skill
    "while True:\n    log('x')",                     # unbounded loop forbidden
    "for _ in range(99999):\n    log('x')",          # exceeds MAX_LOOP_ITERS
    "for _ in range(10 ** 9):\n    log('x')",        # non-literal range arg
    "for i in [1, 2, 3]:\n    log(i)",               # iterable is not range()
    "def f():\n    pass",
    "lambda: 1",
    "[log(x) for x in range(3)]",                    # list comprehension
    "log(x for x in range(3))",                      # generator expression
    "a, b = 1, 2",                                   # tuple-unpacking target
    "x.y = 1",                                        # attribute target
    "del x",
    "with open('x') as f:\n    pass",
    "assert True",
    "raise Exception()",
    "global x",
    "log(*['a'])",                                    # starred arg
    # --- arithmetic / resource-DoS vectors (red-team confirmed) ---
    "x = 2 * 2",                                       # multiplication forbidden
    "result = 1231239 + 1231239",                     # addition forbidden
    "x = 1\nx += 1",                                   # augmented assignment forbidden
    "s = 'a' * 1000000000",                            # string-repeat memory bomb
    "s = 'a'\nfor _ in range(30):\n    s = s + s",     # exponential string-doubling bomb
    "x = 1\nfor _ in range(100):\n    x = x * 2",      # exponential int bomb
    "log('x' + 'y')",                                  # string concat via + (use f-strings instead)
    # --- nested-loop CPU bomb + f-string format-spec allocation ---
    "for _ in range(100):\n    for _ in range(100):\n        for _ in range(2):\n            pass",  # product 20000 > cap
    "log(f'{1:1000000000}')",                          # format spec allocates ~1GB at format time
    "log(f'{1:>10}')",                                 # any f-string format spec is rejected
])
def test_forbidden_plans_rejected(plan):
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SKILLS)


def test_oversized_string_literal_rejected():
    with pytest.raises(PlanValidationError):
        validate_plan("log('%s')" % ("a" * 2000), SKILLS)
