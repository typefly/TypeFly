"""
Trust boundary for LLM-generated plans.

TypeFly asks an LLM for a plan whose ``plan`` field is a *Python program string*
that is then executed to drive a physical robot. That makes this module the most
security-critical code in the project: everything an attacker could reach via a
crafted instruction, a poisoned scene, or a hallucinating model passes through
here before it touches an actuator.

Two independent classes of harm are contained, by two independent controls:

1. **Code escape** — generated Python that breaks out of the skill vocabulary
   (``import os``, ``open(...)``, ``().__class__.__bases__`` reflection, etc.).
   Contained by :func:`validate_plan` (an AST allowlist) + :class:`SkillNamespace`
   (a restricted ``exec`` namespace with only ``SAFE_BUILTINS``).

2. **Action escape** — a *perfectly valid* plan that does something physically
   dangerous: ``move_forward(999)``, ``rotate_right(99999)``, ``delay(99999)``, or
   a huge loop. Contained by :class:`PlanPolicy`, which clamps every skill
   argument and bounds total work.

Why this and not string sanitization or prompt engineering: the prompt layer can
only change the *odds* the model emits something hostile; it can never be relied
on to *stop* it. The AST + policy are the boundary; sanitization is
defense-in-depth.

A note on runtime bounds (see also the controller): Python cannot forcibly kill a
running thread, so a cooperative cancel flag only takes effect when control
returns to us. The validator therefore **forbids ``while`` and unbounded
``for``** (only ``for _ in range(<int literal <= MAX_LOOP_ITERS>)`` is allowed),
**forbids binary arithmetic** (so a bounded loop cannot grow a value into a
memory/CPU bomb — a red-team pass confirmed ``s = s + s`` / ``x = x * 2`` style
attacks), and :class:`PlanPolicy` checks the cancel flag / deadline at every
skill-call boundary. Together those bound total runtime *and* memory by
construction. For a *hard* wall-clock guarantee regardless of code shape (or for
genuinely untrusted deployment), run the plan in a separate process and terminate
it — that is the documented escalation, not implemented here.
"""

from __future__ import annotations

import ast
import json
import re
import time
from typing import Optional


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class PlanValidationError(Exception):
    """Raised when an LLM response cannot be parsed into a safe, runnable plan."""


class PlanCancelled(Exception):
    """Raised inside a running plan when it is cancelled or times out."""


class PlanLimitExceeded(Exception):
    """Raised when a plan exceeds a PlanPolicy resource budget."""


# --------------------------------------------------------------------------- #
# Restricted builtins exposed to executed plans
# --------------------------------------------------------------------------- #
# Deliberately tiny. Notably absent: __import__, eval, exec, compile, open,
# getattr/setattr, globals/locals/vars, type/object, input, breakpoint, bytes —
# anything that enables imports, reflection, or I/O. `range` is required for the
# bounded for-loops the validator permits.
SAFE_BUILTINS: dict = {
    "range": range,
    "len": len,
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "round": round,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "enumerate": enumerate,
}

# Cap on the literal iteration count of any single ``for ... in range(...)``.
MAX_LOOP_ITERS = 100


# --------------------------------------------------------------------------- #
# 1. JSON extraction + shape validation
# --------------------------------------------------------------------------- #
# Case-insensitive, handles ```json ... ``` and bare ``` ... ``` fences.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_plan_json(raw: str) -> dict:
    """Extract and shape-validate the plan JSON from a raw LLM response.

    Returns the parsed object (a ``dict`` with a non-empty string ``plan``), or
    raises :class:`PlanValidationError`. Robust to markdown fences, surrounding
    prose, and JSON-mode (fence-less) output.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PlanValidationError("empty LLM response")

    text = raw.strip()
    match = _FENCE_RE.search(text)
    candidate = match.group(1).strip() if match else text

    obj = _loads_lenient(candidate)
    if not isinstance(obj, dict):
        raise PlanValidationError("plan JSON is not an object")

    plan = obj.get("plan")
    if not isinstance(plan, str) or not plan.strip():
        raise PlanValidationError("plan JSON has no non-empty string 'plan' field")
    return obj


def _loads_lenient(candidate: str) -> object:
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Fallback: grab the outermost {...} span and try again.
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise PlanValidationError("response does not contain a JSON object")
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise PlanValidationError(f"invalid JSON: {exc}") from exc


# --------------------------------------------------------------------------- #
# 2. AST allowlist
# --------------------------------------------------------------------------- #
# Binary arithmetic (+, -, *, /, //, %, **, shifts) is intentionally NOT allowed.
# A red-team pass showed that, combined with bounded loops, arithmetic enables
# in-process memory/CPU bombs that no in-process control can bound: `x = x * 2`
# in a loop -> 2**N, `s = s + s` -> an exponential string, `'a' * 10**9` -> a
# multi-GB allocation. Plans are pure orchestration: skill calls + control flow +
# literals. String formatting that would have used `+` is covered by f-strings.
_ALLOWED_UNARY = (ast.Not, ast.USub, ast.UAdd)
_ALLOWED_CMP = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_ALLOWED_BOOL = (ast.And, ast.Or)
# Cap literal string size so a single huge string literal is not its own DoS.
MAX_STR_LITERAL = 1000


def validate_plan(
    program_str: str,
    allowed_skills,
    max_loop_iters: int = MAX_LOOP_ITERS,
):
    """Validate a plan program string against the allowlist and compile it.

    ``allowed_skills`` is the set/collection of registered skill names (lowercase).
    Returns a compiled code object ready for ``exec`` in a :class:`SkillNamespace`,
    or raises :class:`PlanValidationError`.

    The allowlist permits exactly what real generated plans use — skill calls,
    single-name assignment, ``if/else``, bounded ``for ... in range(<literal>)``,
    ``break``/``continue``, literals, names, arithmetic, comparisons, boolean ops,
    f-strings — and rejects everything else, including ``import``, attribute access
    (which would enable dunder reflection), ``while``, comprehensions, ``lambda``,
    function/class definitions, and calls to anything that is not a known skill.
    """
    allowed = set(allowed_skills)
    try:
        tree = ast.parse(program_str, mode="exec")
    except SyntaxError as exc:
        raise PlanValidationError(f"plan is not valid Python: {exc}") from exc

    _PlanValidator(allowed, max_loop_iters).validate(tree)

    try:
        return compile(tree, "<plan>", "exec")
    except (SyntaxError, ValueError) as exc:  # e.g. 'break' outside loop
        raise PlanValidationError(f"plan failed to compile: {exc}") from exc


class _PlanValidator:
    def __init__(self, allowed_skills: set, max_loop_iters: int):
        self.allowed = allowed_skills
        self.max_loop_iters = max_loop_iters

    def validate(self, module: ast.Module) -> None:
        for stmt in module.body:
            self._stmt(stmt)

    # -- statements --------------------------------------------------------- #
    def _stmt(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr):
            self._expr(node.value)
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise PlanValidationError("only single-name assignment is allowed")
            self._expr(node.value)
        elif isinstance(node, ast.If):
            self._expr(node.test)
            self._body(node.body)
            self._body(node.orelse)
        elif isinstance(node, ast.For):
            self._for(node)
        elif isinstance(node, (ast.Break, ast.Continue, ast.Pass)):
            return
        else:
            raise PlanValidationError(
                f"statement '{type(node).__name__}' is not allowed in a plan"
            )

    def _body(self, stmts) -> None:
        for stmt in stmts:
            self._stmt(stmt)

    def _for(self, node: ast.For) -> None:
        if node.orelse:
            raise PlanValidationError("for/else is not allowed")
        if not isinstance(node.target, ast.Name):
            raise PlanValidationError("loop variable must be a simple name")
        it = node.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
        ):
            raise PlanValidationError(
                "loops must be 'for <name> in range(<int literal>)'"
            )
        if it.keywords or not (1 <= len(it.args) <= 3):
            raise PlanValidationError("range() takes 1-3 positional integer literals")
        literals = [self._int_literal(arg) for arg in it.args]
        if self._range_count(literals) > self.max_loop_iters:
            raise PlanValidationError(
                f"loop exceeds the maximum of {self.max_loop_iters} iterations"
            )
        self._body(node.body)

    @staticmethod
    def _range_count(literals: list) -> int:
        if len(literals) == 1:
            start, stop, step = 0, literals[0], 1
        elif len(literals) == 2:
            start, stop, step = literals[0], literals[1], 1
        else:
            start, stop, step = literals
        if step == 0:
            raise PlanValidationError("range() step must not be zero")
        return max(0, len(range(start, stop, step)))

    @staticmethod
    def _int_literal(node: ast.expr) -> int:
        # bool is a subclass of int; reject it explicitly.
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(
            node.value, bool
        ):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, int)
            and not isinstance(node.operand.value, bool)
        ):
            v = node.operand.value
            return -v if isinstance(node.op, ast.USub) else v
        raise PlanValidationError("range() arguments must be integer literals")

    # -- expressions -------------------------------------------------------- #
    def _expr(self, node: ast.expr) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                if len(node.value) > MAX_STR_LITERAL:
                    raise PlanValidationError(
                        f"string literal exceeds {MAX_STR_LITERAL} characters"
                    )
                return
            if isinstance(node.value, (int, float, bool)) or node.value is None:
                return
            raise PlanValidationError(
                f"literal of type '{type(node.value).__name__}' is not allowed"
            )
        elif isinstance(node, ast.Name):
            return
        elif isinstance(node, ast.Call):
            self._call(node)
        elif isinstance(node, ast.UnaryOp):
            self._require(node.op, _ALLOWED_UNARY, "unary operator")
            self._expr(node.operand)
        elif isinstance(node, ast.BoolOp):
            self._require(node.op, _ALLOWED_BOOL, "boolean operator")
            for value in node.values:
                self._expr(value)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                self._require(op, _ALLOWED_CMP, "comparison")
            self._expr(node.left)
            for comparator in node.comparators:
                self._expr(comparator)
        elif isinstance(node, ast.JoinedStr):  # f-string
            for value in node.values:
                self._expr(value)
        elif isinstance(node, ast.FormattedValue):
            self._expr(node.value)
            if node.format_spec is not None:
                self._expr(node.format_spec)
        else:
            raise PlanValidationError(
                f"expression '{type(node).__name__}' is not allowed in a plan"
            )

    def _call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise PlanValidationError(
                "only direct calls to named skills are allowed (no attribute calls)"
            )
        name = node.func.id
        if name not in self.allowed:
            raise PlanValidationError(f"call to '{name}' is not an allowed skill")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise PlanValidationError("starred arguments are not allowed")
            self._expr(arg)
        for kw in node.keywords:
            if kw.arg is None:
                raise PlanValidationError("**kwargs are not allowed")
            self._expr(kw.value)

    @staticmethod
    def _require(op, allowed, label: str) -> None:
        if not isinstance(op, allowed):
            raise PlanValidationError(f"{label} '{type(op).__name__}' is not allowed")


# --------------------------------------------------------------------------- #
# 3. Restricted exec namespace
# --------------------------------------------------------------------------- #
class SkillNamespace(dict):
    """``globals`` mapping for executing a validated plan.

    Resolves names to (a) registered skills, (b) variables the plan assigns, or
    (c) :data:`SAFE_BUILTINS` — and nothing else. ``exec`` at module scope uses
    ``LOAD_NAME``, which consults ``__getitem__`` for ``dict`` subclasses, so this
    is the runtime half of the trust boundary (the AST validator is the static
    half; either alone would not be sufficient).
    """

    def __init__(self, robot):
        super().__init__()
        # robot.skillset.skills: dict[str, SkillItem]; SkillItem is callable.
        self._skills = robot.skillset.skills
        # Pre-seed builtins so exec's own __builtins__ lookup finds the safe set
        # (and never installs the real builtins module).
        self["__builtins__"] = SAFE_BUILTINS

    def __getitem__(self, key):
        skills = self._skills
        if key in skills:
            return skills[key]
        try:
            return super().__getitem__(key)
        except KeyError:
            pass
        if key in SAFE_BUILTINS:
            return SAFE_BUILTINS[key]
        raise NameError(f"name '{key}' is not defined")


# --------------------------------------------------------------------------- #
# 4. Runtime policy (action-escape containment)
# --------------------------------------------------------------------------- #
class PlanPolicy:
    """Per-plan resource limits enforced at every skill-call boundary.

    The robot wrapper calls :meth:`checkpoint` from its movement/rotation/delay/
    probe skills and :meth:`poll` from internal wait loops. ``checkpoint`` clamps
    physical arguments, counts work against budgets, and (with ``poll``) checks the
    cancel flag and deadline so a plan can be stopped between skill calls.
    """

    def __init__(
        self,
        cancel_event=None,
        deadline: Optional[float] = None,
        max_distance: float = 5.0,
        max_angle: float = 360.0,
        max_delay: float = 30.0,
        max_total_calls: int = 1000,
        max_probe_calls: int = 20,
    ):
        self.cancel_event = cancel_event
        self.deadline = deadline
        self.max_distance = max_distance
        self.max_angle = max_angle
        self.max_delay = max_delay
        self.max_total_calls = max_total_calls
        self.max_probe_calls = max_probe_calls
        self.total_calls = 0
        self.probe_calls = 0

    def poll(self) -> None:
        """Cancel/deadline check only (no counting). For tight internal loops."""
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise PlanCancelled("plan cancelled")
        if self.deadline is not None and time.monotonic() > self.deadline:
            raise PlanCancelled("plan timed out")

    def checkpoint(self, kind: str, value=None):
        """Guard a skill call: cancel-check, count, and clamp ``value``.

        Returns the (possibly clamped) value to pass on to the actuator.
        """
        self.poll()
        self.total_calls += 1
        if self.total_calls > self.max_total_calls:
            raise PlanLimitExceeded(
                f"plan exceeded {self.max_total_calls} skill calls"
            )

        if kind == "move":
            return _clamp(value, -self.max_distance, self.max_distance)
        if kind == "rotate":
            return _clamp(value, -self.max_angle, self.max_angle)
        if kind == "delay":
            return _clamp(value, 0.0, self.max_delay)
        if kind == "probe":
            self.probe_calls += 1
            if self.probe_calls > self.max_probe_calls:
                raise PlanLimitExceeded(
                    f"plan exceeded {self.max_probe_calls} probe calls"
                )
        return value


def _clamp(value, lo: float, hi: float):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return value  # non-numeric; leave for the skill to handle/reject
    return max(lo, min(hi, v))
