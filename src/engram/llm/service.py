"""LLM service for Engram.

This module provides high-level LLM functionality for fact extraction,
summarization, and other AI tasks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import ConfigurationError
from engram.providers.llm import LLMProvider, LLMResponse, get_llm_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# finish_reason values that indicate the model hit its output token cap
# (OpenAI: "length", Anthropic: "max_tokens", some providers: "max_output_tokens")
_TRUNCATION_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}

# Max facts decided in one LLM call. Turns with more facts are chunked into
# bounded concurrent sub-batches so the decision output can't overflow the
# model's token cap, and so no single prompt grows unwieldy.
_MAX_FACTS_PER_DECISION = 12


def _parse_operation_decisions(text: str) -> dict[int, dict[str, Any]]:
    """Parse the batched-decision JSON array into a {fact_number: decision} map.

    Tolerant of code fences and surrounding prose; returns {} on any failure so
    the caller falls back to a safe per-fact ADD.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    start, end = t.find("["), t.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        data = json.loads(t[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}
    out: dict[int, dict[str, Any]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "fact" in item:
                try:
                    out[int(item["fact"])] = item
                except (ValueError, TypeError):
                    continue
    return out


class MemoryOperationType(str, Enum):
    """Types of memory operations.

    Note on DELETE: the caller (Engram.add_conversation) applies both UPDATE
    and DELETE as a supersede — a new active revision that retains the old fact
    in the same lineage for audit/history. DELETE therefore means "replace the
    contradicted fact", not "hard-remove the row".
    """

    ADD = "ADD"  # Create new memory
    UPDATE = "UPDATE"  # Augment existing memory with new info
    DELETE = "DELETE"  # Replace a contradicted memory (supersede, history kept)
    NOOP = "NOOP"  # No operation needed (duplicate)


@dataclass
class MemoryOperation:
    """Represents an intelligent memory operation to perform.

    Attributes:
        operation: Type of operation (ADD, UPDATE, DELETE, NOOP).
        content: The fact/content to store or the merged content for UPDATE.
        target_id: Memory ID to update/delete (for UPDATE/DELETE operations).
        original_fact: The original extracted fact.
        reason: Why this operation was chosen.
    """

    operation: MemoryOperationType
    content: str
    target_id: str | None = None
    original_fact: str = ""
    reason: str = ""
    memory_type: str = "semantic"


@dataclass
class ExtractionResult:
    """Result of intelligent fact extraction and processing.

    Attributes:
        facts: List of extracted atomic facts.
        operations: List of memory operations to execute.
        summary: Optional conversation summary for future context.
    """

    facts: list[str] = field(default_factory=list)
    operations: list[MemoryOperation] = field(default_factory=list)
    summary: str | None = None


class LLMService:
    """High-level LLM service for AI tasks.

    This service provides:
    - Fact extraction from conversations
    - Text summarization
    - Question answering
    - General completions

    Example:
        # Create from settings
        service = LLMService.from_settings()

        # Or with explicit provider
        service = LLMService.from_provider("openai", api_key="sk-...")

        # Extract facts
        facts = await service.extract_facts(
            user_message="I love pizza and live in NYC",
            assistant_response="That's great! NYC has amazing pizza.",
        )

        # General completion
        response = await service.complete("What is 2 + 2?")
    """

    def __init__(self, provider: LLMProvider) -> None:
        """Initialize the LLM service.

        Args:
            provider: The LLM provider instance to use.
        """
        self._provider = provider
        logger.info(
            f"Initialized LLMService with {provider.__class__.__name__} "
            f"(model={provider.model})"
        )

    @property
    def model(self) -> str:
        """Get the model name."""
        return self._provider.model

    @property
    def provider(self) -> LLMProvider:
        """Get the underlying provider."""
        return self._provider

    @classmethod
    def from_provider(
        cls,
        provider_name: str,
        **kwargs: Any,
    ) -> LLMService:
        """Create an LLMService with a specific provider.

        Args:
            provider_name: Name of the LLM provider.
            **kwargs: Provider-specific configuration.

        Returns:
            Configured LLMService.

        Example:
            # OpenAI
            service = LLMService.from_provider(
                "openai",
                api_key="sk-...",
                model="gpt-4o-mini",
            )

            # Anthropic
            service = LLMService.from_provider(
                "anthropic",
                api_key="sk-ant-...",
                model="claude-haiku-4-5-20251001",
            )

            # Ollama (local)
            service = LLMService.from_provider(
                "ollama",
                model="llama3.2",
            )
        """
        provider = get_llm_provider(provider_name, **kwargs)
        return cls(provider=provider)

    @classmethod
    def from_settings(
        cls,
        settings: EngramSettings | None = None,
    ) -> LLMService | None:
        """Create an LLMService from settings.

        Uses the provider registry to create the appropriate provider
        based on the ENGRAM_LLM_PROVIDER setting.

        Args:
            settings: Engram settings. If None, loads from environment.

        Returns:
            Configured LLMService, or None if llm_provider is not set.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        settings = settings or get_settings()

        if not settings.llm_provider:
            logger.info("LLM provider not configured, LLM features disabled")
            return None

        provider_name = settings.llm_provider
        provider_kwargs = settings.get_llm_provider_kwargs()

        try:
            provider = get_llm_provider(provider_name, **provider_kwargs)
        except KeyError as e:
            raise ConfigurationError(str(e)) from e
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create LLM provider '{provider_name}': {e}"
            ) from e

        return cls(provider=provider)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a text completion.

        Args:
            prompt: The user prompt.
            system: Optional system message.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Returns:
            The generated text.
        """
        return await self._provider.complete_text(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def complete_full(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a completion with full response metadata.

        Args:
            messages: List of conversation messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Provider-specific parameters.

        Returns:
            Full LLM response with metadata.
        """
        return await self._provider.complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    async def extract_facts(
        self,
        user_message: str,
        assistant_response: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        conversation_summary: str | None = None,
    ) -> list[str]:
        """Extract atomic facts from a conversation exchange.

        Uses a comprehensive prompt with conversation context for better extraction.

        Args:
            user_message: Current user message.
            assistant_response: Current assistant response.
            conversation_history: Recent messages for temporal context (last 10 recommended).
            conversation_summary: Optional semantic summary of conversation history.

        Returns:
            List of extracted atomic facts (empty if none found).

        Example:
            facts = await service.extract_facts(
                user_message="I'm meeting Sarah at Southeast Bank at 3pm",
                assistant_response="Have a good meeting!",
                conversation_history=[
                    {"role": "user", "content": "My name is Nafiz"},
                    {"role": "assistant", "content": "Nice to meet you!"},
                ],
            )
            # Returns: [
            #     "User has a meeting at 3pm",
            #     "User is meeting someone named Sarah",
            #     "User's bank is Southeast Bank",
            # ]
        """
        # Build context sections
        context_parts = []

        # Add conversation summary if provided
        if conversation_summary:
            context_parts.append(f"Conversation Summary:\n{conversation_summary}")

        # Add recent message history for temporal context
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-10:]:  # Last 10 messages
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")[:500]  # Truncate long messages
                history_lines.append(f"{role}: {content}")
            if history_lines:
                context_parts.append("Recent Context:\n" + "\n".join(history_lines))

        context_block = "\n\n".join(context_parts) + "\n\n" if context_parts else ""

        extraction_prompt = f"""<task>
Extract ALL atomic facts about the user that are asserted by the USER MESSAGE below.
An atomic fact is a single, self-contained piece of information that can stand alone.
</task>

<source_of_truth priority="highest">
The user message is the only source of new or corrected memory.
Use assistant text only as context for disambiguation, never as a source of
new facts. If a fact appears only in the assistant response, do not extract it.
If the user asks a recall, comparison, timeline, summary, or history question,
that question is not itself a new memory unless the user also states a new fact
or correction.
</source_of_truth>

<context>
{context_block if context_block else "<none/>"}
</context>

<current_exchange>
<user>{user_message}</user>
<assistant_context>{assistant_response}</assistant_context>
</current_exchange>

<categories description="Extract ALL that apply">
<category name="identity">name, age, birthday, gender, nationality, location, where they live</category>
<category name="work">job title, company name, role, what they work on, technologies used</category>
<category name="projects">SPECIFIC project names, what they're building, project status (v1, done, in progress)</category>
<category name="education">university, degree, field of study, courses</category>
<category name="relationships">family members, friends, colleagues, pets (INCLUDE NAMES)</category>
<category name="preferences">likes, dislikes, favorites (food, music, hobbies)</category>
<category name="schedule">appointments, meetings, plans, events (include times/dates/locations)</category>
<category name="goals">aspirations, things they want to do, future plans</category>
<category name="possessions">devices, vehicles, items they own</category>
<category name="reference">fictional/test reference codes, labels, confirmation numbers, safe-note labels, non-secret recovery hints</category>
<category name="other">any other SPECIFIC personal information</category>
</categories>

<rules priority="critical">
<rule id="1" name="be_specific">
Extract exact names, project names, company names, technologies.
<example type="bad">User has a favorite project</example>
<example type="good">User is building AI memory layer for GenAI apps</example>
</rule>

<rule id="2" name="separate_facts">
Extract EVERY distinct fact separately - don't combine.
<example type="bad">User works at X and is building Y</example>
<example type="good">Two facts: "User works at X" AND "User is building Y"</example>
</rule>

<rule id="3" name="include_status">
Include current status when mentioned.
<example input="I already done the v1">User completed version 1 of their AI memory layer project</example>
</rule>

<rule id="4" name="user_subject">Use "User" as subject: "User's birthday is October 24"</rule>

<rule id="5" name="precise_names">Be precise with names: "User's friend Sarah" not "User has a friend"</rule>

<rule id="6" name="implicit_facts">
Extract implicit facts.
<example input="I'm late for CS class">User studies Computer Science</example>
</rule>

<rule id="7" name="possessive_chains">
Preserve possessive chains.
<example input="my girlfriend's name is X">User's girlfriend's name is X</example>
</rule>

<rule id="8" name="no_vague_facts">
NEVER return vague facts like "User has a project (not mentioned)" - either extract the specific name or don't include it.
</rule>

<rule id="9" name="third_party_attribution" priority="highest">
CRITICAL: The User is the person chatting (saying "I", "my", "we"). When the User mentions OTHER people by name, those are NOT the User!
<example type="bad" input="Amy quit her job">User quit her job ❌ WRONG - Amy is not the User!</example>
<example type="good" input="Amy quit her job">User's wife Amy quit her job ✓</example>
<example type="bad" input="Lily started school">User started school ❌ WRONG</example>
<example type="good" input="Lily started school">User's daughter Lily started school ✓</example>
<example type="bad" input="Karim moved to London">User moved to London ❌ WRONG</example>
<example type="good" input="Karim moved to London">User's brother Karim moved to London ✓</example>
Always include the RELATIONSHIP when referring to third parties: "User's wife Amy", "User's brother Karim", "User's child Lucas".
</rule>

<rule id="10" name="relationship_context">
Use conversation history to identify relationships. If history says "my wife Amy", then later "Amy did X" means "User's wife Amy did X".
</rule>

<rule id="11" name="reference_codes_and_sensitive_values">
Extract fictional/test reference codes, safe-note labels, recovery hints, confirmation
numbers, and user instructions about how they should be shared when the user
explicitly asks you to remember them. Do not extract real passwords, API keys,
private keys, full access tokens, or payment card numbers unless the user clearly
labels them as fictional test data.
<example type="good" input="fictional: the recovery hint ends with 47-Kilo">User's fictional recovery hint ends with 47-Kilo</example>
<example type="good" input="the safe-note label is Violet">User's safe-note label is Violet</example>
</rule>

<rule id="12" name="assistant_is_not_memory_source" priority="highest">
Do not extract assistant-restated memories. Assistant answers are often recalled
from existing memory and must not rewrite user-authored memories.
<example type="bad" input_user="What was my meeting change?" input_assistant="It changed from 3 PM to 10 PM">User's meeting changed from 3 PM to 10 PM ❌ WRONG - assistant-only restatement</example>
<example type="good" input_user="What was my meeting change?" input_assistant="It changed from 3 PM to 10 PM">NONE</example>
<example type="good" input_user="Actually move my meeting from 3 PM to 10 PM" input_assistant="Noted">User changed their meeting from 3 PM to 10 PM</example>
</rule>
</rules>

<output_format>
Return one fact per line.
Return "NONE" ONLY if truly no extractable facts.
</output_format>

<exclusions>
<exclude>Vague facts with "(not mentioned)" or "(unknown)"</exclude>
<exclude>Generic statements without specifics</exclude>
<exclude>Assistant's opinions</exclude>
<exclude>Facts stated only by the assistant</exclude>
<exclude>Recall/history/summary questions with no new user-stated fact</exclude>
<exclude>Hypotheticals</exclude>
</exclusions>"""

        try:
            response = await self._provider.complete(
                [{"role": "user", "content": extraction_prompt}],
                max_tokens=1000,
                temperature=0,
            )

            text = (response.content or "").strip()
            if not text or text.upper() == "NONE":
                return []

            facts = []
            for line in text.split("\n"):
                # Clean up various bullet formats
                fact = line.strip().lstrip("-•*·").strip()
                # Remove numbering like "1.", "1)", etc.
                if fact and len(fact) > 1 and fact[0].isdigit():
                    fact = fact.lstrip("0123456789.)").strip()

                # Only include meaningful facts
                if fact and len(fact) > 5 and fact.upper() != "NONE":
                    facts.append(fact)

            # If output hit the token cap, the last line is likely cut
            # mid-sentence; storing a partial fact is worse than missing one.
            finish = (response.finish_reason or "").lower()
            if finish in _TRUNCATION_FINISH_REASONS and facts:
                logger.warning(
                    f"Fact extraction output truncated (finish_reason="
                    f"{response.finish_reason}); dropping last partial fact"
                )
                facts = facts[:-1]

            return facts

        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []

    async def extract_document_facts(
        self,
        text: str,
        *,
        kind: str | None = None,
        heading: str | None = None,
        max_facts: int = 8,
    ) -> list[str]:
        """Extract atomic facts from a document chunk (not a conversation).

        The conversation prompt (extract_facts) phrases everything as facts
        about "the User", which is wrong for ingested documents, contracts,
        and specs. This prompt extracts the document's own statements
        verbatim-faithfully.

        Args:
            text: The document chunk text.
            kind: Optional chunk classification (requirement, constraint, ...).
            heading: Optional section heading for context.
            max_facts: Upper bound on returned facts.

        Returns:
            List of extracted facts (empty if none found).
        """
        context_lines = []
        if heading:
            context_lines.append(f"Section: {heading}")
        if kind:
            context_lines.append(f"Chunk type: {kind}")
        context_block = "\n".join(context_lines) if context_lines else "<none/>"

        prompt = f"""<task>
Extract up to {max_facts} atomic facts from the document excerpt below.
An atomic fact is one self-contained statement that can stand alone.
</task>

<context>
{context_block}
</context>

<document_excerpt>
{text}
</document_excerpt>

<rules>
<rule>Preserve exact names, numbers, dates, thresholds, and identifiers.</rule>
<rule>State requirements, constraints, decisions, and deadlines as the document states them.</rule>
<rule>Do NOT phrase facts as being about "the User" — this is a document, not a chat.</rule>
<rule>One fact per line. No numbering, no commentary.</rule>
<rule>Return "NONE" only if the excerpt contains no extractable statements.</rule>
</rules>"""

        try:
            response = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0,
            )
            content = (response.content or "").strip()
            if not content or content.upper() == "NONE":
                return []

            facts = []
            for line in content.split("\n"):
                fact = line.strip().lstrip("-•*·").strip()
                if fact and len(fact) > 1 and fact[0].isdigit():
                    fact = fact.lstrip("0123456789.)").strip()
                if fact and len(fact) > 5 and fact.upper() != "NONE":
                    facts.append(fact)

            finish = (response.finish_reason or "").lower()
            if finish in _TRUNCATION_FINISH_REASONS and facts:
                logger.warning(
                    f"Document fact extraction truncated (finish_reason="
                    f"{response.finish_reason}); dropping last partial fact"
                )
                facts = facts[:-1]

            return facts[:max_facts]

        except Exception as e:
            logger.warning(f"Document fact extraction failed: {e}")
            return []

    async def classify_facts(self, facts: list[str]) -> list[str]:
        """Classify each fact into a memory type via one batched LLM call.

        Returns a list aligned with ``facts``. Any unparseable entry defaults
        to "semantic".
        """
        if not facts:
            return []

        numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
        prompt = f"""<task>
Classify each numbered fact into exactly one memory type.
</task>

<types>
<type name="profile">Identity, location, health, relationships, durable user facts. ("User's name is Sara", "User is allergic to shellfish")</type>
<type name="preference">Stable user preferences or communication style. ("User prefers concise bullets")</type>
<type name="project">Project/product facts, owners, codenames, launch facts, metrics. ("Atlas Checkout rollback owner is Priya")</type>
<type name="task">Task-specific requirements, pending work, acceptance criteria. ("The task requires end-to-end tests")</type>
<type name="constraint">Hard rules, repo constraints, safety limits, deadlines. ("Never schedule Friday meetings after 2 PM")</type>
<type name="decision">Explicit decisions or corrections. ("The p95 target changed to 160ms")</type>
<type name="tool_result">Tool outputs, measurements, test results, observations. ("Load test p95 was 172ms")</type>
<type name="semantic">Generic durable fact that does not fit a more specific type.</type>
<type name="episodic">A dated or time-bound event that happened — WHAT happened. ("User attended a concert on May 9", "User moved to Berlin last week")</type>
<type name="procedural">A behavioral rule for how the assistant should act — HOW to behave. ("Always reply formally", "Never mention pricing")</type>
</types>

<facts>
{numbered}
</facts>

<output_format>
Return one line per fact in order: "<number>: <type>".
Use only: profile, preference, project, task, constraint, decision, tool_result, semantic, episodic, procedural.
Default to semantic if unsure.
</output_format>"""

        # Longest-first fixed order so compound answers like "task/decision"
        # resolve deterministically (set iteration order would be random).
        valid = (
            "tool_result",
            "constraint",
            "preference",
            "procedural",
            "decision",
            "episodic",
            "semantic",
            "profile",
            "project",
            "task",
        )
        types: list[str] = ["semantic"] * len(facts)
        try:
            response = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0,
            )
            finish = (response.finish_reason or "").lower()
            if finish in _TRUNCATION_FINISH_REASONS:
                logger.warning(
                    f"Fact classification output truncated (finish_reason="
                    f"{response.finish_reason}); missing entries default to semantic"
                )
            for line in (response.content or "").strip().split("\n"):
                line = line.strip().lower()
                if ":" not in line:
                    continue
                num_str, type_str = line.split(":", 1)
                num_str = num_str.strip().lstrip("#").strip()
                chosen = next((t for t in valid if t in type_str), None)
                if chosen and num_str.isdigit():
                    idx = int(num_str) - 1
                    if 0 <= idx < len(facts):
                        types[idx] = chosen
            return types
        except Exception as e:
            logger.warning(f"Fact classification failed: {e}")
            return types

    async def expand_query(self, query: str, n_queries: int = 4) -> list[str]:
        """Rewrite a query into several search variants for higher recall (HyDE).

        Returns up to ``n_queries`` alternative phrasings that together widen
        recall against a memory store. Returns [] on failure (caller should fall
        back to the original query).
        """
        if n_queries < 1:
            return []

        prompt = f"""<task>
Rewrite the search query into {n_queries} alternative queries that together
maximize recall against a store of facts about a user. Cover different angles:
a comprehensive third-person restatement, the key entities/names, the
action/target, and literal nouns or numbers. Include indirectly relevant
constraints and preferences: food queries should search for allergies,
avoidances, dietary restrictions, and restaurant preferences; scheduling
queries should search for time boundaries and preferred call windows; update or
"old plan" queries should search for cancelled, superseded, replaced, moved, and
"no longer" facts.
</task>

<query>{query}</query>

<output_format>
Return exactly {n_queries} queries, one per line, no numbering or commentary.
</output_format>"""

        try:
            response = await self._provider.complete_text(
                prompt=prompt, max_tokens=200, temperature=0
            )
            variants: list[str] = []
            for line in response.strip().split("\n"):
                v = line.strip().lstrip("-•*").strip()
                if v and v[0].isdigit():
                    v = v.lstrip("0123456789.)").strip()
                if v and len(v) > 2:
                    variants.append(v)
            return variants[:n_queries]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return []

    async def evaluate_memory_operation(
        self,
        new_fact: str,
        existing_memories: list[tuple[str, str]],  # List of (memory_id, content)
    ) -> MemoryOperation:
        """Evaluate what operation to perform for a new fact.

        Determines whether to ADD, UPDATE, DELETE, or NOOP based on existing memories.

        Args:
            new_fact: The newly extracted fact.
            existing_memories: List of (memory_id, content) tuples for similar memories.

        Returns:
            MemoryOperation with the appropriate action.
        """
        if not existing_memories:
            return MemoryOperation(
                operation=MemoryOperationType.ADD,
                content=new_fact,
                original_fact=new_fact,
                reason="No similar memories exist",
            )

        memories_text = "\n".join(
            f"{i + 1}. {content}" for i, (_, content) in enumerate(existing_memories)
        )

        prompt = f"""<task>
Compare this new fact against existing memories and decide the appropriate memory operation.
</task>

<new_fact>{new_fact}</new_fact>

<existing_memories>
{memories_text}
</existing_memories>

<decision_rules priority_order="true">
<rule operation="DELETE" priority="1">
Use when new fact CONTRADICTS/REPLACES an existing memory about the SAME person and attribute.
<criteria>
<item>Same person's job changed (quit old job, joined new company)</item>
<item>Same person's location changed (moved from X to Y)</item>
<item>Same person's status changed (was X, now Y)</item>
<item>Correction of previous information (had ADHD → is autistic)</item>
</criteria>
<examples>
<example>"Amy joined Doctors Without Borders" contradicts "Amy is a doctor at Johns Hopkins" = DELETE</example>
<example>"User switched to Bank B" contradicts "User banks at Bank A" = DELETE</example>
<example>"User now lives in NYC" contradicts "User lives in Dhaka" = DELETE</example>
<example>"Lucas is autistic" contradicts "Lucas has ADHD" = DELETE (correction!)</example>
<example>"Amy quit her job at Johns Hopkins" contradicts "Amy works at Johns Hopkins" = DELETE</example>
</examples>
</rule>

<rule operation="ADD" priority="2">
Use when fact contains NEW INFORMATION not in any existing memory.
<criteria>
<item>Different topic/category (e.g., job vs project vs location)</item>
<item>Same person but different attribute (job vs age vs hobby)</item>
<item>Different person entirely</item>
</criteria>
<example>
"User is building AI memory layer" vs "User works at AskTuring" = ADD (different info!)
</example>
</rule>

<rule operation="NOOP" priority="3">
Use ONLY when new fact is SEMANTICALLY IDENTICAL to an existing memory.
<criteria>
<item>Must be SAME topic AND SAME information</item>
<item>Only minor wording differences</item>
</criteria>
<examples>
<example>"User's LinkedIn is X" ≈ "User's LinkedIn profile is X" = NOOP</example>
<example>"User has cats Luna and Milo" ≈ "User has 2 cats named Luna and Milo" = NOOP</example>
</examples>
</rule>

<rule operation="UPDATE" priority="4">
Use when new fact ADDS DETAIL to existing memory (merge them).
<example>"User's sister Nadia" + "Nadia lives in Toronto" = UPDATE with merged</example>
</rule>
</decision_rules>

<common_mistakes>
<mistake type="avoid">NOOP for job changes - if someone changed jobs, DELETE the old job fact!</mistake>
<mistake type="avoid">NOOP for corrections - if a fact is being corrected, DELETE the wrong one!</mistake>
<mistake type="avoid">NOOP for loosely related facts (job vs project) - these should be ADD</mistake>
<mistake type="correct">DELETE when same attribute (job, location, status) changes for same person</mistake>
</common_mistakes>

<default_behavior>
Default to ADD if unsure - it's better to store extra than lose information.
</default_behavior>

<output_format strict="true">
Respond in this EXACT format (4 lines):
OPERATION: [ADD|UPDATE|DELETE|NOOP]
TARGET: [memory number if UPDATE or DELETE, otherwise none]
MERGED: [complete merged content if UPDATE, otherwise none]
REASON: [brief explanation]
</output_format>"""

        try:
            response = await self._provider.complete_text(
                prompt=prompt,
                max_tokens=300,
                temperature=0,
            )

            op_type = MemoryOperationType.ADD
            target_idx: int | None = None
            merged_content: str | None = None
            reason = ""

            for line in response.strip().split("\n"):
                line = line.strip()
                if line.upper().startswith("OPERATION:"):
                    op_str = line.split(":", 1)[1].strip().upper().split()[0]
                    if "UPDATE" in op_str:
                        op_type = MemoryOperationType.UPDATE
                    elif "DELETE" in op_str:
                        op_type = MemoryOperationType.DELETE
                    elif "NOOP" in op_str:
                        op_type = MemoryOperationType.NOOP
                    else:
                        op_type = MemoryOperationType.ADD
                elif line.upper().startswith("TARGET:"):
                    target_str = line.split(":", 1)[1].strip()
                    # Extract first number
                    nums = "".join(c for c in target_str if c.isdigit())
                    if nums:
                        target_idx = int(nums) - 1
                elif line.upper().startswith("MERGED:"):
                    merged_content = line.split(":", 1)[1].strip()
                    if merged_content.lower() == "none":
                        merged_content = None
                elif line.upper().startswith("REASON:"):
                    reason = line.split(":", 1)[1].strip()

            # Determine final content and target_id
            target_id: str | None = None
            if target_idx is not None and 0 <= target_idx < len(existing_memories):
                target_id = existing_memories[target_idx][0]

            # UPDATE/DELETE without a resolvable target would silently drop
            # the fact downstream. Fall back to ADD (the prompt's own default:
            # better to store extra than lose information).
            if (
                op_type in (MemoryOperationType.UPDATE, MemoryOperationType.DELETE)
                and target_id is None
            ):
                logger.debug(
                    f"{op_type.value} operation had unresolvable target; "
                    f"falling back to ADD for fact: {new_fact[:80]}"
                )
                return MemoryOperation(
                    operation=MemoryOperationType.ADD,
                    content=new_fact,
                    original_fact=new_fact,
                    reason=f"{op_type.value} target unresolvable; stored as new"
                    + (f" ({reason})" if reason else ""),
                )

            # Determine content based on operation
            if op_type == MemoryOperationType.UPDATE:
                content = merged_content if merged_content else new_fact
            elif op_type == MemoryOperationType.DELETE:
                content = new_fact  # The new fact that replaces the deleted one
            else:
                content = new_fact

            return MemoryOperation(
                operation=op_type,
                content=content,
                target_id=target_id,
                original_fact=new_fact,
                reason=reason,
            )

        except Exception as e:
            logger.warning(f"Memory operation evaluation failed: {e}")
            return MemoryOperation(
                operation=MemoryOperationType.ADD,
                content=new_fact,
                original_fact=new_fact,
                reason=f"Fallback due to error: {e}",
            )

    def _operation_from_decision(
        self,
        fact: str,
        candidates: list[tuple[str, str]],
        decision: dict[str, Any] | None,
    ) -> MemoryOperation:
        """Build a MemoryOperation from one parsed batched-decision object.

        Applies the same target resolution and ADD-fallback safety as
        evaluate_memory_operation: a missing decision, or an UPDATE/DELETE with
        an unresolvable target, degrades to ADD so a fact is never silently
        dropped.
        """
        if not decision:
            return MemoryOperation(
                operation=MemoryOperationType.ADD,
                content=fact,
                original_fact=fact,
                reason="No decision returned; stored as new",
            )

        op_raw = str(decision.get("operation", "ADD")).upper()
        if "DELETE" in op_raw:
            op_type = MemoryOperationType.DELETE
        elif "UPDATE" in op_raw:
            op_type = MemoryOperationType.UPDATE
        elif "NOOP" in op_raw:
            op_type = MemoryOperationType.NOOP
        else:
            op_type = MemoryOperationType.ADD

        reason = str(decision.get("reason") or "")

        target_id: str | None = None
        target = decision.get("target")
        if target is not None:
            try:
                idx = int(target) - 1
                if 0 <= idx < len(candidates):
                    target_id = candidates[idx][0]
            except (ValueError, TypeError):
                target_id = None

        # UPDATE/DELETE without a resolvable target would drop the fact; the
        # safe default (per the prompt) is to store it as new.
        if (
            op_type in (MemoryOperationType.UPDATE, MemoryOperationType.DELETE)
            and target_id is None
        ):
            logger.debug(
                f"{op_type.value} operation had unresolvable target; "
                f"falling back to ADD for fact: {fact[:80]}"
            )
            return MemoryOperation(
                operation=MemoryOperationType.ADD,
                content=fact,
                original_fact=fact,
                reason=f"{op_type.value} target unresolvable; stored as new"
                + (f" ({reason})" if reason else ""),
            )

        if op_type == MemoryOperationType.UPDATE:
            merged = decision.get("merged")
            content = (
                str(merged)
                if merged and str(merged).strip().lower() != "none"
                else fact
            )
        else:
            content = fact

        return MemoryOperation(
            operation=op_type,
            content=content,
            target_id=target_id,
            original_fact=fact,
            reason=reason,
        )

    async def decide_memory_operations(
        self,
        items: list[tuple[str, list[tuple[str, str]]]],
        *,
        max_facts_per_call: int = _MAX_FACTS_PER_DECISION,
    ) -> list[MemoryOperation]:
        """Batched form of evaluate_memory_operation: decide ADD/UPDATE/DELETE/
        NOOP for many facts with as few LLM calls as possible.

        Each item is ``(fact, [(memory_id, content), ...])`` — the fact and its
        own related existing memories. The per-fact loop in process_for_memory
        otherwise costs one LLM round-trip per fact; this collapses them into a
        single call. Once there are more than ``max_facts_per_call`` facts the
        work is chunked into bounded concurrent sub-batches, so a large turn
        can never overflow the model's output token cap or produce an unwieldy
        single prompt.

        Args:
            items: Facts paired with their candidate memories (each fact should
                have at least one candidate; facts with none are pure ADDs and
                need no LLM decision).
            max_facts_per_call: Upper bound on facts decided per LLM call.

        Returns:
            Operations aligned 1:1 with ``items``. Any fact whose decision is
            missing or unparseable falls back to ADD.
        """
        if not items:
            return []
        if len(items) <= max_facts_per_call:
            return await self._decide_batch(items)

        # Many facts: split into bounded batches, decide them concurrently, and
        # stitch the results back together in the original order.
        batches = [
            items[i : i + max_facts_per_call]
            for i in range(0, len(items), max_facts_per_call)
        ]
        results = await asyncio.gather(*(self._decide_batch(b) for b in batches))
        return [op for batch_ops in results for op in batch_ops]

    async def _decide_batch(
        self,
        items: list[tuple[str, list[tuple[str, str]]]],
    ) -> list[MemoryOperation]:
        """Decide operations for one bounded batch of facts in a single call."""
        if not items:
            return []

        blocks: list[str] = []
        for n, (fact, candidates) in enumerate(items, start=1):
            cand_lines = "\n".join(
                f"    {j}. {content}"
                for j, (_id, content) in enumerate(candidates, start=1)
            )
            blocks.append(
                f'<fact n="{n}">{fact}</fact>\n'
                f'<existing_memories_for_fact n="{n}">\n{cand_lines}\n'
                f"</existing_memories_for_fact>"
            )
        facts_block = "\n\n".join(blocks)

        prompt = f"""<task>
For EACH numbered fact, compare it against ONLY that fact's own existing
memories and decide exactly one operation: ADD, UPDATE, DELETE, or NOOP.
</task>

<decision_rules priority_order="true">
<rule operation="DELETE" priority="1">New fact CONTRADICTS/REPLACES an existing memory about the SAME person and attribute (job/location/status change, or a correction).</rule>
<rule operation="ADD" priority="2">Fact contains NEW INFORMATION not in any existing memory (different topic, attribute, or person).</rule>
<rule operation="NOOP" priority="3">Fact is SEMANTICALLY IDENTICAL to an existing memory (same topic AND same information; only wording differs).</rule>
<rule operation="UPDATE" priority="4">Fact ADDS DETAIL to an existing memory (merge them into one).</rule>
</decision_rules>

<common_mistakes>
<mistake>NOOP for a job/location/status change — that is DELETE.</mistake>
<mistake>NOOP for a correction — that is DELETE.</mistake>
<mistake>NOOP for loosely related facts (job vs project) — that is ADD.</mistake>
</common_mistakes>

<default>If unsure, choose ADD — better to store extra than lose information.</default>

<facts>
{facts_block}
</facts>

<output_format strict="true">
Return ONLY a JSON array, one object per fact, in fact-number order:
[{{"fact": 1, "operation": "ADD|UPDATE|DELETE|NOOP", "target": <existing memory number for UPDATE/DELETE, else null>, "merged": "<merged text for UPDATE, else null>", "reason": "<brief>"}}]
"target" refers to the numbered existing memory listed under THAT fact only.
No prose, no code fences.
</output_format>"""

        decisions: dict[int, dict[str, Any]] = {}
        try:
            response = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=max(300, len(items) * 140),
                temperature=0,
            )
            finish = (response.finish_reason or "").lower()
            if finish in _TRUNCATION_FINISH_REASONS:
                logger.warning(
                    "Batched memory operation output truncated (finish_reason="
                    f"{response.finish_reason}); missing facts default to ADD"
                )
            decisions = _parse_operation_decisions(response.content or "")
        except Exception as e:
            logger.warning(f"Batched memory operation decision failed: {e}")

        return [
            self._operation_from_decision(fact, candidates, decisions.get(n))
            for n, (fact, candidates) in enumerate(items, start=1)
        ]

    async def process_for_memory(
        self,
        user_message: str,
        assistant_response: str,
        existing_memories: list[tuple[Any, ...]],  # (id, content, score[, semantic])
        *,
        conversation_history: list[dict[str, str]] | None = None,
        conversation_summary: str | None = None,
        similarity_threshold: float = 0.50,  # Only consider highly related memories
        duplicate_threshold: float = 0.92,  # Only skip truly identical facts
        retrieve_for_fact: Callable[[str], Awaitable[list[tuple[Any, ...]]]]
        | None = None,
        classify_types: bool = False,
    ) -> ExtractionResult:
        """Complete intelligent fact extraction and memory operation pipeline.

        This is the main entry point for memory processing. It:
        1. Extracts atomic facts from the conversation exchange
        2. For each fact, evaluates against existing memories
        3. Determines the appropriate operation (ADD/UPDATE/DELETE/NOOP)
        4. Returns operations ready to execute

        Args:
            user_message: Current user message.
            assistant_response: Current assistant response.
            existing_memories: List of (memory_id, content, score) or
                (memory_id, content, score, semantic_score) tuples from a
                single search on the user message. The optional 4th element
                is raw cosine similarity, used for the duplicate check (the
                combined score includes decay and is unsuitable for it).
                Used only when retrieve_for_fact is not provided.
            conversation_history: Recent messages for context (last 10 recommended).
            conversation_summary: Optional semantic summary.
            similarity_threshold: Min similarity to consider memories related (default 0.3).
            duplicate_threshold: Similarity above which fact is considered duplicate (default 0.92).
            retrieve_for_fact: Optional async callback that, given an extracted
                fact, returns its own (id, content, score) candidates. When set,
                dedup/consolidation candidates are fetched per fact instead of
                reusing the message-level existing_memories — the correct
                behavior when facts span multiple topics.

        Returns:
            ExtractionResult with facts and operations to execute.

        Example:
            # Search for similar memories first
            similar = await engram.search(query=user_message, limit=10)
            existing = [(m.memory.memory_id, m.memory.content, m.score) for m in similar]

            # Process the exchange
            result = await llm.process_for_memory(
                user_message="I switched to BRAC Bank last week",
                assistant_response="Good choice!",
                existing_memories=existing,
                conversation_history=history,
            )

            # Execute operations
            for op in result.operations:
                if op.operation == MemoryOperationType.ADD:
                    await engram.add(content=op.content, ...)
                elif op.operation in (
                    MemoryOperationType.UPDATE,
                    MemoryOperationType.DELETE,
                ):
                    # Both apply as a supersede: a new active revision that
                    # keeps the old fact in the lineage for audit/history.
                    await engram.revise(op.target_id, content=op.content)
        """
        result = ExtractionResult()

        # Phase 1: Extract atomic facts
        facts = await self.extract_facts(
            user_message,
            assistant_response,
            conversation_history=conversation_history,
            conversation_summary=conversation_summary,
        )
        result.facts = facts

        if not facts:
            return result

        # Phase 2: retrieve candidates for every fact, then decide all
        # operations in a SINGLE batched LLM call. Retrieval runs concurrently
        # and the previous one-LLM-call-per-fact loop is replaced by one
        # decide_memory_operations() call, so cost is ~constant in fact count.
        if retrieve_for_fact is not None:
            candidate_lists = list(
                await asyncio.gather(*(retrieve_for_fact(fact) for fact in facts))
            )
        else:
            candidate_lists = [existing_memories for _ in facts]

        # Candidates are (id, content, score) or (id, content, score,
        # semantic_score). The combined score ranks relevance; the duplicate
        # check needs raw cosine similarity, because the hybrid combined score
        # includes time decay and sags below the threshold within hours even
        # for identical facts.
        def _similarity(cand: tuple) -> float:  # type: ignore[type-arg]
            return float(cand[3]) if len(cand) > 3 else float(cand[2])

        # operations[i] holds fact i's resolved op (1:1 with facts, in order).
        operations: list[MemoryOperation | None] = [None] * len(facts)
        # Facts that need an LLM decision: (fact_index, fact, relevant_memories).
        to_decide: list[tuple[int, str, list[tuple[str, str]]]] = []

        for i, (fact, candidates) in enumerate(
            zip(facts, candidate_lists, strict=True)
        ):
            # Quick duplicate guard (no LLM): raw cosine >= duplicate_threshold.
            dup = next(
                (c for c in candidates if _similarity(c) >= duplicate_threshold),
                None,
            )
            if dup is not None:
                operations[i] = MemoryOperation(
                    operation=MemoryOperationType.NOOP,
                    content=fact,
                    original_fact=fact,
                    reason=f"Duplicate of existing memory: {dup[1][:50]}...",
                )
                continue

            relevant = [
                (c[0], c[1]) for c in candidates if c[2] >= similarity_threshold
            ]
            if not relevant:
                # Nothing to compare against -> a new fact, no LLM needed.
                operations[i] = MemoryOperation(
                    operation=MemoryOperationType.ADD,
                    content=fact,
                    original_fact=fact,
                    reason="No similar memories exist",
                )
                continue

            to_decide.append((i, fact, relevant))

        if to_decide:
            decided = await self.decide_memory_operations(
                [(fact, relevant) for _i, fact, relevant in to_decide]
            )
            for (i, _fact, _relevant), op in zip(to_decide, decided, strict=True):
                operations[i] = op

        # Every slot is filled above; the ADD fallback is purely defensive.
        result.operations = [
            op
            or MemoryOperation(
                operation=MemoryOperationType.ADD, content=fact, original_fact=fact
            )
            for op, fact in zip(operations, facts, strict=True)
        ]

        # Optionally tag each operation with a cognitive memory type. Operations
        # are 1:1 with extracted facts, in order.
        if classify_types and result.operations:
            types = await self.classify_facts(facts)
            for op, mem_type in zip(result.operations, types, strict=False):
                op.memory_type = mem_type

        return result

    async def summarize(
        self,
        text: str,
        *,
        max_length: int = 100,
        style: str = "concise",
    ) -> str:
        """Summarize text.

        Args:
            text: Text to summarize.
            max_length: Approximate max length of summary in words.
            style: Summary style ('concise', 'detailed', 'bullet').

        Returns:
            The summary.
        """
        style_instructions = {
            "concise": f"Summarize in {max_length} words or less.",
            "detailed": f"Provide a detailed summary in {max_length} words.",
            "bullet": f"Summarize as bullet points ({max_length} words max).",
        }

        prompt = f"""<task>
<instruction>{style_instructions.get(style, style_instructions["concise"])}</instruction>
<style>{style}</style>
<max_words>{max_length}</max_words>
</task>

<input>
{text}
</input>

<output>
Provide the summary below:
</output>"""

        return await self._provider.complete_text(
            prompt=prompt,
            max_tokens=max_length * 2,  # Rough token estimate
            temperature=0.3,
        )

    async def update_conversation_summary(
        self,
        previous_summary: str | None,
        user_message: str,
        assistant_response: str,
        *,
        max_length: int = 200,
        style: str = "concise",
    ) -> str:
        """Roll a conversation summary forward with a new exchange.

        Updates the previous summary in place rather than re-summarizing from
        scratch, preserving information across many turns (see the context
        compression design). Used by Engram.add_conversation() to keep a
        compact per-session summary.

        Args:
            previous_summary: The prior summary, or None for the first exchange.
            user_message: The latest user message.
            assistant_response: The latest assistant response.
            max_length: Approximate max length of the summary in words.
            style: ``"concise"`` (default, free-form) or ``"structured"`` — the
                Goal / Constraints / Progress / Decisions / Next Steps /
                Critical Context template, iteratively updated for better
                long-conversation retention.

        Returns:
            The updated summary text.
        """
        prev = previous_summary.strip() if previous_summary else ""
        update_or_write = (
            "Update the existing summary below to incorporate the new exchange"
            if prev
            else "Write a summary of the exchange below"
        )

        if style == "structured":
            prompt = f"""<task>
Maintain a running, STRUCTURED summary of a conversation so it can be resumed
later without losing durable information. {update_or_write}, keeping it under
{max_length} words total. Preserve durable facts, goals, constraints,
preferences, and decisions; drop small talk. Move items between sections as
they evolve and remove obsolete entries.
</task>

<template>
## Goal
[What the user is trying to accomplish]

## Constraints & Preferences
[Durable preferences, constraints, standing instructions]

## Progress
[What has been established or done so far]

## Key Decisions
[Important decisions or corrections, and why]

## Next Steps
[What should happen next, if stated]

## Critical Context
[Specific values, names, dates, identifiers worth keeping verbatim]
</template>

<existing_summary>
{prev if prev else "<none/>"}
</existing_summary>

<new_exchange>
<user>{user_message}</user>
<assistant>{assistant_response}</assistant>
</new_exchange>

<output>
Return only the updated structured summary using the template headings (omit a
section if it has no content). No preamble.
</output>"""
        else:
            prompt = f"""<task>
Maintain a running summary of a conversation. {update_or_write}.
Keep it under {max_length} words. Preserve durable facts, goals, and decisions; drop small talk.
</task>

<existing_summary>
{prev if prev else "<none/>"}
</existing_summary>

<new_exchange>
<user>{user_message}</user>
<assistant>{assistant_response}</assistant>
</new_exchange>

<output>
Return only the updated summary text, no preamble.
</output>"""

        summary = await self._provider.complete_text(
            prompt=prompt,
            max_tokens=max_length * 2,
            temperature=0.3,
        )
        return summary.strip()
