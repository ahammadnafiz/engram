"""LLM service for Engram.

This module provides high-level LLM functionality for fact extraction,
summarization, and other AI tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from engram.core.config import EngramSettings, get_settings
from engram.core.exceptions import ConfigurationError
from engram.providers.llm import LLMProvider, LLMResponse, get_llm_provider

logger = logging.getLogger(__name__)


class MemoryOperationType(str, Enum):
    """Types of memory operations (Mem0-style)."""
    ADD = "ADD"        # Create new memory
    UPDATE = "UPDATE"  # Augment existing memory with new info
    DELETE = "DELETE"  # Remove contradicted memory
    NOOP = "NOOP"      # No operation needed (duplicate)


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
    ) -> "LLMService":
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
                model="claude-3-haiku-20240307",
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
    ) -> "LLMService | None":
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
        Based on the Mem0 research paper's incremental extraction approach.
        
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
                context_parts.append(f"Recent Context:\n" + "\n".join(history_lines))
        
        context_block = "\n\n".join(context_parts) + "\n\n" if context_parts else ""
        
        extraction_prompt = f"""{context_block}Current Exchange:
User: {user_message}
Assistant: {assistant_response}

Extract ALL atomic facts about the user from this exchange. An atomic fact is a single, 
self-contained piece of information that can stand alone.

Categories to extract:
- Identity: name, age, birthday, gender, nationality, location
- Professional: job title, company, university, field of study, skills, projects
- Relationships: family members, friends, colleagues, pets (include names)
- Preferences: likes, dislikes, favorites (food, music, colors, hobbies)
- Schedule: appointments, meetings, plans, events (include times/dates/locations)
- Goals: aspirations, things they want to do, projects planned
- Possessions: devices, vehicles, items they own
- Health: conditions, allergies, dietary restrictions
- Contact: email, phone, social media handles
- Any other specific personal information

Rules:
1. Extract EVERY distinct fact - don't combine multiple facts into one
2. Each fact must be self-contained and understandable without context
3. Include specific details: names, times, dates, places, numbers
4. Use "User" as the subject (e.g., "User's birthday is October 24")
5. Be precise - "User's friend Sarah" not just "User has a friend"
6. Extract implicit facts (e.g., "I'm late for my CS class" → "User studies Computer Science")
7. CRITICAL: Preserve relationships - when user mentions "my X's name" or "my X is Y", the fact is about USER'S X, not about USER directly. Always maintain the possessive chain accurately.

Return one fact per line. Return "NONE" only if there are truly no facts.
Do NOT include:
- Generic statements without specific information
- Assistant's opinions or suggestions
- Hypotheticals or maybes
- Misattributed facts (e.g., attributing someone else's name/property to the user)"""

        try:
            response = await self._provider.complete_text(
                prompt=extraction_prompt,
                max_tokens=500,
                temperature=0,
            )
            
            response = response.strip()
            if not response or response.upper() == "NONE":
                return []
            
            facts = []
            for line in response.split("\n"):
                # Clean up various bullet formats
                fact = line.strip().lstrip("-•*·").strip()
                # Remove numbering like "1.", "1)", etc.
                if fact and len(fact) > 1 and fact[0].isdigit():
                    fact = fact.lstrip("0123456789.)").strip()
                
                # Only include meaningful facts
                if fact and len(fact) > 5 and fact.upper() != "NONE":
                    facts.append(fact)
            
            return facts
            
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
            return []

    async def evaluate_memory_operation(
        self,
        new_fact: str,
        existing_memories: list[tuple[str, str]],  # List of (memory_id, content)
    ) -> MemoryOperation:
        """Evaluate what operation to perform for a new fact (Mem0-style update phase).
        
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
            f"{i+1}. {content}" for i, (_, content) in enumerate(existing_memories)
        )
        
        prompt = f"""Compare this new fact against existing memories and decide the operation.

New Fact: {new_fact}

Existing Memories:
{memories_text}

DECISION RULES:
1. NOOP - Use when new fact says THE SAME THING as an existing memory (even if worded differently)
   Example: "User's LinkedIn is X" and "User's LinkedIn profile is X" = NOOP
   
2. DELETE - Use when new fact CONTRADICTS/REPLACES an existing memory
   Example: "User switched to Bank B" contradicts "User banks at Bank A" = DELETE #n
   Example: "User now studies Y" contradicts "User studies X" = DELETE #n
   
3. UPDATE - Use when new fact ADDS DETAIL to an existing memory (merge them)
   Example: "User's sister Nadia" + "Nadia is in Toronto" = UPDATE with merged info
   
4. ADD - Use ONLY when the fact is completely new information

CRITICAL: Look for semantic equivalence, not just word matching!
- "LinkedIn is X" ≈ "LinkedIn profile is X" = SAME (NOOP)
- "has 2 cats Luna and Milo" ≈ "has cats named Luna and Milo" = SAME (NOOP)
- "switched to Y" means old X is WRONG = DELETE old, the new fact replaces it

Respond in this EXACT format (4 lines):
OPERATION: <ADD|UPDATE|DELETE|NOOP>
TARGET: <memory number if UPDATE or DELETE, otherwise none>
MERGED: <complete merged content if UPDATE, otherwise none>
REASON: <brief explanation>"""

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

    async def process_for_memory(
        self,
        user_message: str,
        assistant_response: str,
        existing_memories: list[tuple[str, str, float]],  # (id, content, similarity_score)
        *,
        conversation_history: list[dict[str, str]] | None = None,
        conversation_summary: str | None = None,
        similarity_threshold: float = 0.35,
        duplicate_threshold: float = 0.85,  # Lowered to catch semantic duplicates
    ) -> ExtractionResult:
        """Complete intelligent fact extraction and memory operation pipeline (Mem0-style).
        
        This is the main entry point for memory processing. It:
        1. Extracts atomic facts from the conversation exchange
        2. For each fact, evaluates against existing memories
        3. Determines the appropriate operation (ADD/UPDATE/DELETE/NOOP)
        4. Returns operations ready to execute
        
        Args:
            user_message: Current user message.
            assistant_response: Current assistant response.
            existing_memories: List of (memory_id, content, similarity_score) from semantic search.
            conversation_history: Recent messages for context (last 10 recommended).
            conversation_summary: Optional semantic summary.
            similarity_threshold: Min similarity to consider memories related (default 0.3).
            duplicate_threshold: Similarity above which fact is considered duplicate (default 0.92).
            
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
                elif op.operation == MemoryOperationType.UPDATE:
                    await engram.update(op.target_id, content=op.content)
                elif op.operation == MemoryOperationType.DELETE:
                    await engram.forget(op.target_id)
                    await engram.add(content=op.content, ...)  # Add replacement
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
        
        # Phase 2: Process each fact
        for fact in facts:
            # Find relevant existing memories for this specific fact
            relevant_memories: list[tuple[str, str]] = []
            
            for mem_id, content, score in existing_memories:
                if score >= similarity_threshold:
                    relevant_memories.append((mem_id, content))
            
            # Quick duplicate check - if very high similarity to any memory, skip
            is_duplicate = False
            for mem_id, content, score in existing_memories:
                if score >= duplicate_threshold:
                    # Check if this specific fact is the duplicate
                    # (the existing_memories might match the user message, not the fact)
                    # Do a quick content comparison
                    fact_lower = fact.lower()
                    content_lower = content.lower()
                    if (fact_lower in content_lower or 
                        content_lower in fact_lower or
                        self._similar_content(fact, content)):
                        is_duplicate = True
                        result.operations.append(MemoryOperation(
                            operation=MemoryOperationType.NOOP,
                            content=fact,
                            original_fact=fact,
                            reason=f"Duplicate of existing memory: {content[:50]}...",
                        ))
                        break
            
            if is_duplicate:
                continue
            
            # Evaluate operation
            operation = await self.evaluate_memory_operation(fact, relevant_memories)
            result.operations.append(operation)
        
        return result

    def _similar_content(self, a: str, b: str) -> bool:
        """Quick check if two strings have similar content (without embeddings)."""
        # Normalize
        a_lower = a.lower()
        b_lower = b.lower()
        
        # Extract key entities (names, numbers, proper nouns pattern)
        import re
        
        def extract_key_terms(text: str) -> set[str]:
            words = set(text.split())
            # Remove common words
            stopwords = {
                "the", "a", "an", "is", "are", "was", "were", "user", "user's", 
                "has", "have", "had", "at", "in", "on", "to", "for", "of", "and",
                "their", "they", "them", "this", "that", "with", "from", "by",
                "named", "called", "known", "as", "also", "now", "currently",
            }
            words -= stopwords
            return words
        
        a_words = extract_key_terms(a_lower)
        b_words = extract_key_terms(b_lower)
        
        if not a_words or not b_words:
            return False
        
        # Check for key entity overlap (names, places, etc.)
        # These are likely proper nouns - capitalized in original
        a_entities = set(w for w in a.split() if w[0].isupper() and len(w) > 2)
        b_entities = set(w for w in b.split() if w[0].isupper() and len(w) > 2)
        
        # If same entities mentioned, likely related
        entity_overlap = len(a_entities & b_entities)
        if entity_overlap >= 1 and len(a_entities | b_entities) <= 3:
            # Same entity, check if same topic
            topic_words = {"linkedin", "email", "phone", "birthday", "bank", "job", 
                          "sister", "brother", "friend", "girlfriend", "boyfriend",
                          "cat", "dog", "pet", "study", "work", "live", "from"}
            a_topics = a_words & topic_words
            b_topics = b_words & topic_words
            if a_topics & b_topics:
                return True
        
        # Jaccard similarity with lower threshold for longer texts
        intersection = len(a_words & b_words)
        union = len(a_words | b_words)
        
        if union == 0:
            return False
        
        jaccard = intersection / union
        
        # More lenient for shorter facts
        threshold = 0.5 if len(a_words) < 5 or len(b_words) < 5 else 0.4
        
        return jaccard > threshold
    
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
        
        prompt = f"""{style_instructions.get(style, style_instructions['concise'])}

Text to summarize:
{text}

Summary:"""

        return await self._provider.complete_text(
            prompt=prompt,
            max_tokens=max_length * 2,  # Rough token estimate
            temperature=0.3,
        )
    
    async def answer_question(
        self,
        question: str,
        context: str,
    ) -> str:
        """Answer a question given context.
        
        Args:
            question: The question to answer.
            context: Context/background information.
            
        Returns:
            The answer.
        """
        prompt = f"""Answer the question based on the provided context.
If the answer cannot be found in the context, say "I don't know based on the provided information."

Context:
{context}

Question: {question}

Answer:"""

        return await self._provider.complete_text(
            prompt=prompt,
            temperature=0.3,
        )

