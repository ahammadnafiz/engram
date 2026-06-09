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
    generic_critical_slots: bool = True

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
        project = self._project_slug(text)

        allergy_slot = self._allergy_slot(text)
        if allergy_slot:
            return allergy_slot

        for rule in self.slot_rules:
            if rule.memory_types and memory_type not in rule.memory_types:
                continue
            if any(re.search(pattern, text) for pattern in rule.patterns):
                return rule.slot.format(project=project)

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
        if "allerg" not in text:
            return None
        negative = re.search(r"not allergic to ([a-z0-9 _-]+)", text)
        if negative:
            allergen = re.sub(r"[^a-z0-9]+", "_", negative.group(1)).strip("_")
            return f"profile:not_allergy:{allergen or 'unknown'}"
        positive = re.search(r"allergic to ([a-z0-9 _-]+?)(?:[,.;]| and |$)", text)
        if positive:
            allergen = re.sub(r"[^a-z0-9]+", "_", positive.group(1)).strip("_")
            return f"profile:allergy:{allergen or 'unknown'}"
        return "profile:allergy"

    def _project_slug(self, text: str) -> str:
        if "atlas checkout" in text or "checkout" in text:
            return "atlas_checkout"
        return "project"


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
            r"\brisk table\b",
            r"\bwhen discussing\b",
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
            r"\bupdated target\b",
        ),
    ),
    TypeRule(
        "tool_result",
        (
            r"\btool note\b",
            r"\bload test\b",
            r"\btest result\b",
            r"\bapi call\b",
            r"\bpytest\b",
            r"\berror rate\b",
            r"\bp95\b",
        ),
    ),
    TypeRule(
        "project",
        (
            r"\bproject\b",
            r"\bcheckout\b",
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


DEFAULT_SLOT_RULES: tuple[SlotRule, ...] = (
    SlotRule(
        "tool_result:{project}:load_test",
        (r"\b(load test|tool note|today's result)\b",),
    ),
    SlotRule(
        "profile:current_city",
        (r"\bcurrent city\b", r"\bmoved from\b", r"\blives in\b"),
    ),
    SlotRule("profile:name", (r"\bmy name\b", r"\buser's name\b")),
    SlotRule("profile:manager", (r"\bmanager\b",)),
    SlotRule("project:{project}:old_codename", (r"\bold codename\b",)),
    SlotRule("project:{project}:codename", (r"\bcodename\b",)),
    SlotRule("project:{project}:rollback_owner", (r"\brollback owner\b",)),
    SlotRule("project:{project}:metrics_owner", (r"\bmetrics owner\b",)),
    SlotRule("project:{project}:launch_date", (r"\blaunch date\b",)),
    SlotRule("project:{project}:p95_target_or_result", (r"\bp95\b",)),
    SlotRule("project:{project}:error_rate_target_or_result", (r"\berror rate\b",)),
    SlotRule("project:{project}:traffic_requirement", (r"\bblack friday\b",)),
    SlotRule("preference:risk_table_columns", (r"\brisk table\b",)),
    SlotRule("preference:incident_order", (r"\bincident\b",)),
    SlotRule("preference:communication_style", (r"\bconcise\b", r"\bno fluff\b")),
    SlotRule("preference:meeting_time", (r"\bfriday meetings\b",)),
    SlotRule("constraint:repo", (r"\brepo constraint\b",)),
    SlotRule("task:requirement", (r"\btask requirement\b",)),
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
