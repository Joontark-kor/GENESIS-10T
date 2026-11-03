"""Safety-check interface. GENESIS defines the contract; implementations live with each application.

The core ships no screening rules. An application provides a class with
``check_input`` and ``check_output`` and the core calls it around the model:

    check = load_safety_check("my_app.safety:Checker")   # or GENESIS_SAFETY_CHECK env var
    result = guarded_chat(check, ask_model, messages)

``ask_model`` is any function from messages to reply text, for example
``lambda m: EndpointClient(url).chat("fin", m).text``. If the input check does
not allow the request, the model is never called. If the output check asks
for a replacement, the replacement is returned instead of the model's reply.
Every verdict is kept in the result so the application can log it.

Without a configured check, ``AllowAll`` is used and the result says so,
so "no safety check" is always visible rather than silent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import importlib
import os
from typing import Callable, Protocol, runtime_checkable


class Action(str, Enum):
    ALLOW = "allow"        # go on
    BLOCK = "block"        # stop; return ``replacement`` (or nothing) to the user
    REPLACE = "replace"    # output only: send ``replacement`` instead of the model reply
    HANDOFF = "handoff"    # stop and pass the conversation to a person; the app decides how


@dataclass(frozen=True)
class Verdict:
    action: Action = Action.ALLOW
    reasons: tuple[str, ...] = ()
    replacement: str | None = None
    labels: dict = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.action is Action.ALLOW


@runtime_checkable
class SafetyCheck(Protocol):
    name: str

    def check_input(self, messages: list[dict]) -> Verdict: ...

    def check_output(self, messages: list[dict], reply: str) -> Verdict: ...


class AllowAll:
    """Default when nothing is configured: allows everything and says that no check ran."""

    name = "none"

    def check_input(self, messages: list[dict]) -> Verdict:
        return Verdict(reasons=("no safety check configured",))

    def check_output(self, messages: list[dict], reply: str) -> Verdict:
        return Verdict(reasons=("no safety check configured",))


def load_safety_check(spec: str | None = None) -> SafetyCheck:
    """``package.module:ClassName`` (instantiated with no arguments), else ``GENESIS_SAFETY_CHECK``, else AllowAll."""
    spec = spec or os.environ.get("GENESIS_SAFETY_CHECK")
    if not spec:
        return AllowAll()
    if ":" not in spec:
        raise ValueError(f"safety check must be module:ClassName, got {spec!r}")
    module, attr = spec.split(":", 1)
    check = getattr(importlib.import_module(module), attr)()
    if not isinstance(check, SafetyCheck):
        raise TypeError(f"{spec} must define name, check_input and check_output")
    return check


@dataclass
class GuardedReply:
    text: str | None
    called_model: bool
    input_verdict: Verdict
    output_verdict: Verdict | None
    check_name: str

    @property
    def action(self) -> Action:
        if not self.input_verdict.allowed:
            return self.input_verdict.action
        return self.output_verdict.action if self.output_verdict else Action.ALLOW


def guarded_chat(check: SafetyCheck, ask_model: Callable[[list[dict]], str],
                 messages: list[dict]) -> GuardedReply:
    before = check.check_input(messages)
    if not before.allowed:
        return GuardedReply(before.replacement, False, before, None, check.name)
    reply = ask_model(messages)
    after = check.check_output(messages, reply)
    text = reply if after.allowed else after.replacement
    return GuardedReply(text, True, before, after, check.name)
