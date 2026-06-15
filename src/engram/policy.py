"""Configurable memory policy for typing, critical recall, and conflicts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.core._types import AgentId, MemoryType, Metadata, UserId


TYPE_LABELS: dict[str, str] = {
    "semantic": "Semantic facts",
    "episodic": "Events",
    "procedural": "Procedural rules",
    "profile": "Profile facts",
    "project": "Project facts",
    "task": "Task requirements",
    "preference": "Preferences",
    "constraint": "Constraints",
    "decision": "Decisions",
    "tool_result": "Tool results",
}


@dataclass(frozen=True)
class TypeRule:
    """Rule that maps memory text/metadata to a memory type."""

    memory_type: MemoryType
    patterns: tuple[str, ...]
    metadata_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class SlotRule:
    """Rule that maps critical facts to deterministic conflict slots."""

    slot: str
    patterns: tuple[str, ...]
    memory_types: tuple[MemoryType, ...] = ()


@dataclass(frozen=True)
class MemoryPolicy:
    """Policy object used by Engram to govern memory behavior.

    Applications can supply their own policy to tune critical facts, typing,
    and conflict slots for a domain such as personal assistants, coding agents,
    legal review, research, or support.
    """

    name: str = "default"
    critical_memory_types: tuple[MemoryType, ...] = (
        "profile",
        "preference",
        "constraint",
        "project",
        "task",
        "decision",
    )
    type_rules: tuple[TypeRule, ...] = field(default_factory=tuple)
    slot_rules: tuple[SlotRule, ...] = field(default_factory=tuple)
    # Off by default: a content-digest slot can never match a *reworded*
    # contradiction, so it cannot supersede anything real — it only adds a
    # junk conflict_key and an extra supersede query on every add. Explicit
    # slot_rules are the supported conflict mechanism.
    generic_critical_slots: bool = False

    def infer_type(
        self,
        content: str,
        memory_type: MemoryType,
        metadata: Metadata | None = None,
    ) -> MemoryType:
        if memory_type not in {"semantic", "episodic", "procedural"}:
            return memory_type

        metadata = metadata or {}
        category = str(metadata.get("category") or metadata.get("type") or "").lower()
        if category in TYPE_LABELS:
            return category  # type: ignore[return-value]

        text = content.lower()
        for rule in self.type_rules:
            if category and category in rule.metadata_values:
                return rule.memory_type
            if any(re.search(pattern, text) for pattern in rule.patterns):
                return rule.memory_type
        return memory_type

    def critical_slot(self, content: str, memory_type: MemoryType) -> str | None:
        text = content.lower()

        allergy_slot = self._allergy_slot(text)
        if allergy_slot:
            return allergy_slot

        for rule in self.slot_rules:
            if rule.memory_types and memory_type not in rule.memory_types:
                continue
            if any(re.search(pattern, text) for pattern in rule.patterns):
                return rule.slot

        if self.generic_critical_slots and memory_type in self.critical_memory_types:
            digest = hashlib.sha1(content.lower().encode()).hexdigest()[:12]
            return f"{memory_type}:generic:{digest}"
        return None

    def apply_metadata(
        self,
        *,
        content: str,
        agent_id: AgentId,
        user_id: UserId | None,
        memory_type: MemoryType,
        metadata: Metadata | None = None,
    ) -> tuple[MemoryType, Metadata]:
        merged: Metadata = dict(metadata or {})
        inferred_type = self.infer_type(content, memory_type, merged)
        slot = str(
            merged.get("critical_slot")
            or self.critical_slot(content, inferred_type)
            or ""
        )

        merged.setdefault("status", "active")
        merged.setdefault("version", 1)
        merged.setdefault("memory_type", inferred_type)
        if inferred_type in self.critical_memory_types or slot:
            merged.setdefault("critical", True)
        if slot:
            scope_user = user_id or "*"
            merged.setdefault("critical_slot", slot)
            merged.setdefault("conflict_key", f"{agent_id}:{scope_user}:{slot}")
        return inferred_type, merged

    def _allergy_slot(self, text: str) -> str | None:
        """Slot allergy statements by allergen.

        Positive ("allergic to X") and negative ("not allergic to X")
        statements share the same slot so a correction supersedes the
        contradicted fact instead of coexisting with it. Compound allergens
        ("shellfish and peanuts") are kept whole.
        """
        if "allerg" not in text:
            return None
        match = re.search(r"allergic to ([a-z0-9 _-]+?)(?:[,.;]|$)", text)
        if match:
            allergen = re.sub(r"[^a-z0-9]+", "_", match.group(1)).strip("_")
            return f"profile:allergy:{allergen or 'unknown'}"
        return "profile:allergy"


# Generic, domain-neutral typing rules for a general-purpose assistant.
# Domain-specific vocabulary (ops metrics, project codenames, legal/coding
# terms) belongs in a named preset (LEGAL_MEMORY_POLICY, CODING_AGENT_MEMORY_POLICY)
# or a caller-supplied MemoryPolicy, not in the default.
DEFAULT_TYPE_RULES: tuple[TypeRule, ...] = (
    TypeRule(
        "profile",
        (
            r"\ballerg",
            r"\bmy name\b",
            r"\buser's name\b",
            r"\bmanager\b",
            r"\blives in\b",
            r"\bmoved from\b",
            r"\bcurrent city\b",
        ),
    ),
    TypeRule(
        "preference",
        (
            r"\bprefer",
            r"\bpreference\b",
            r"\bno fluff\b",
            r"\bconcise\b",
        ),
    ),
    TypeRule(
        "constraint",
        (
            r"\bmust\b",
            r"\bnever\b",
            r"\balways\b",
            r"\bconstraint\b",
            r"\bdeadline\b",
            r"\bdo not\b",
            r"\bdon't\b",
        ),
    ),
    TypeRule(
        "decision",
        (
            r"\bdecided\b",
            r"\bdecision\b",
            r"\bchanged\b",
            r"\bcorrection\b",
        ),
    ),
    TypeRule(
        "tool_result",
        (
            r"\btest result\b",
            r"\bapi call\b",
        ),
    ),
    TypeRule(
        "project",
        (
            r"\bproject\b",
            r"\blaunch\b",
            r"\bcodename\b",
            r"\bowner\b",
            r"\btarget\b",
        ),
    ),
    TypeRule(
        "task",
        (
            r"\btask\b",
            r"\brequirement\b",
            r"\btodo\b",
            r"\bpending\b",
            r"\bcompleted\b",
        ),
    ),
)


# Generic conflict slots only: durable personal facts a correction should
# supersede. Allergy slotting is handled in code (_allergy_slot), independent
# of these rules. Project/ops/domain slots live in the named presets below.
DEFAULT_SLOT_RULES: tuple[SlotRule, ...] = (
    SlotRule(
        "profile:current_city",
        (r"\bcurrent city\b", r"\bmoved from\b", r"\blives in\b"),
    ),
    SlotRule("profile:name", (r"\bmy name\b", r"\buser's name\b")),
    SlotRule("profile:manager", (r"\bmanager\b",)),
    SlotRule("preference:communication_style", (r"\bconcise\b", r"\bno fluff\b")),
)


DEFAULT_MEMORY_POLICY = MemoryPolicy(
    name="default",
    type_rules=DEFAULT_TYPE_RULES,
    slot_rules=DEFAULT_SLOT_RULES,
)


LEGAL_MEMORY_POLICY = MemoryPolicy(
    name="legal",
    critical_memory_types=(
        "profile",
        "preference",
        "constraint",
        "project",
        "task",
        "decision",
        "tool_result",
    ),
    type_rules=(
        TypeRule(
            "constraint",
            (
                r"\bshall\b",
                r"\bmust\b",
                r"\bliability\b",
                r"\bindemn",
                r"\bconfidential",
                r"\bgoverning law\b",
                r"\bdeadline\b",
            ),
        ),
        TypeRule(
            "decision",
            (
                r"\bdecided\b",
                r"\bapproved\b",
                r"\brejected\b",
                r"\bredline\b",
                r"\bflag\b",
            ),
        ),
        TypeRule(
            "task",
            (
                r"\breview\b",
                r"\brisk table\b",
                r"\banswer with citations\b",
                r"\bsource chunk\b",
            ),
        ),
        TypeRule(
            "tool_result",
            (r"\bextraction result\b", r"\bocr\b", r"\bdocument parser\b"),
        ),
        *DEFAULT_TYPE_RULES,
    ),
    slot_rules=(
        SlotRule(
            "legal:citation_requirement", (r"\bcitations?\b", r"\bsource chunks?\b")
        ),
        SlotRule("legal:audit_logs", (r"\baudit logs?\b",)),
        SlotRule("legal:liability", (r"\bliability\b", r"\bcap liability\b")),
        SlotRule("legal:indemnity", (r"\bindemn",)),
        SlotRule("legal:confidentiality", (r"\bconfidential",)),
        SlotRule("legal:deadline", (r"\bdeadline\b",)),
        *DEFAULT_SLOT_RULES,
    ),
)


CODING_AGENT_MEMORY_POLICY = MemoryPolicy(
    name="coding_agent",
    type_rules=(
        TypeRule(
            "constraint",
            (
                r"\bdo not edit\b",
                r"\bnever revert\b",
                r"\brepo constraint\b",
                r"\bsandbox\b",
                r"\bapproval\b",
            ),
        ),
        TypeRule("task", (r"\bimplement\b", r"\btest\b", r"\bfix\b", r"\bship\b")),
        TypeRule(
            "tool_result",
            (r"\bpytest\b", r"\bnpm\b", r"\bbuild failed\b", r"\bexit code\b"),
        ),
        TypeRule("decision", (r"\bdecided\b", r"\buse\b", r"\bchanged approach\b")),
        *DEFAULT_TYPE_RULES,
    ),
    slot_rules=(
        SlotRule(
            "coding:repo_constraint",
            (r"\brepo constraint\b", r"\bnever revert\b", r"\bdo not edit\b"),
        ),
        SlotRule(
            "coding:test_result", (r"\bpytest\b", r"\btest result\b", r"\bexit code\b")
        ),
        SlotRule(
            "coding:implementation_decision", (r"\bdecided\b", r"\bchanged approach\b")
        ),
        *DEFAULT_SLOT_RULES,
    ),
)


POLICY_PRESETS = {
    "default": DEFAULT_MEMORY_POLICY,
    "legal": LEGAL_MEMORY_POLICY,
    "coding_agent": CODING_AGENT_MEMORY_POLICY,
}


def get_memory_policy(policy: str | MemoryPolicy | None = None) -> MemoryPolicy:
    """Resolve a named or explicit memory policy."""
    if policy is None:
        return DEFAULT_MEMORY_POLICY
    if isinstance(policy, MemoryPolicy):
        return policy
    try:
        return POLICY_PRESETS[policy]
    except KeyError as exc:
        raise ValueError(f"Unknown memory policy: {policy!r}") from exc
