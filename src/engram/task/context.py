"""Budgeted context assembly for long-running task memory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from engram.task.models import AgentEvent, ContextBuildOptions, ContextBuildResult

if TYPE_CHECKING:
    from engram.core._types import AgentId, UserId
    from engram.graph.models import TraversalResult
    from engram.memory.models import SearchResult
    from engram.task.manager import TaskMemoryManager

SearchFn = Callable[..., Awaitable[list["SearchResult"]]]
TraverseFn = Callable[..., Awaitable[list["TraversalResult"]]]
TokenCounter = Callable[[str], int]


class ContextBuilder:
    """Build deterministic prompt context from task memory sources."""

    def __init__(
        self,
        task_manager: TaskMemoryManager,
        search_fn: SearchFn,
        traverse_fn: TraverseFn | None = None,
    ) -> None:
        self._tasks = task_manager
        self._search = search_fn
        self._traverse = traverse_fn

    async def build(
        self,
        *,
        task_run_id: str,
        agent_id: AgentId,
        user_id: UserId | None = None,
        options: ContextBuildOptions,
        token_counter: TokenCounter | None = None,
    ) -> ContextBuildResult:
        """Build a budgeted task context block."""
        count = token_counter or (lambda text: max(1, len(text) // 4))
        task = await self._tasks.get_task(task_run_id)
        events = await self._tasks.list_events(
            task_run_id=task_run_id,
            limit=options.recent_event_limit,
        )
        checkpoints = await self._tasks.list_checkpoints(
            task_run_id,
            limit=options.checkpoint_limit,
        )

        query = options.query or task.goal
        memories = await self._search(
            query,
            agent_id,
            user_id=user_id,
            limit=options.memory_limit,
        )

        graph_lines: list[str] = []
        if options.include_graph and self._traverse is not None:
            seen: set[str] = set()
            for result in memories[:3]:
                for neighbor in await self._traverse(
                    result.memory.memory_id,
                    max_depth=1,
                    direction="any",
                    limit=5,
                ):
                    if neighbor.memory_id in seen:
                        continue
                    seen.add(neighbor.memory_id)
                    graph_lines.append(
                        f"- {neighbor.content} (relation={neighbor.relation_type}, depth={neighbor.depth})"
                    )

        raw_sections = {
            "Task": self._render_task(task.goal, task.status, task.outcome),
            "Current State": self._render_checkpoints(checkpoints[:1]),
            "Recent Events": self._render_events(events),
            "Checkpoints": self._render_checkpoints(checkpoints[1:]),
            "Relevant Memories": "\n".join(f"- {r.memory.content}" for r in memories),
            "Decisions and Constraints": self._render_decisions(checkpoints, events),
            "Artifacts and Tool Results": self._render_artifacts(events),
            "Related Memory Graph": "\n".join(graph_lines),
        }

        budgets = self._section_budgets(options.max_tokens)
        kept: dict[str, str] = {}
        for section, body in raw_sections.items():
            body = body.strip()
            if not body:
                continue
            kept_body = self._trim_lines(
                body,
                budgets.get(section, options.max_tokens),
                count,
            )
            if kept_body:
                kept[section] = kept_body

        text = "\n\n".join(f"## {name}\n{body}" for name, body in kept.items())
        return ContextBuildResult(
            text=text,
            sections=kept,
            token_estimate=count(text),
            metadata={
                "task_run_id": task_run_id,
                "events": len(events),
                "memories": len(memories),
                "checkpoints": len(checkpoints),
            },
        )

    def _section_budgets(self, total: int) -> dict[str, int]:
        fractions = {
            "Task": 0.10,
            "Current State": 0.15,
            "Recent Events": 0.20,
            "Checkpoints": 0.10,
            "Relevant Memories": 0.20,
            "Decisions and Constraints": 0.10,
            "Artifacts and Tool Results": 0.10,
            "Related Memory Graph": 0.05,
        }
        return {name: max(50, int(total * frac)) for name, frac in fractions.items()}

    def _trim_lines(self, text: str, budget: int, count: TokenCounter) -> str:
        used = 0
        kept: list[str] = []
        for line in text.splitlines():
            line_cost = count(line)
            if used + line_cost > budget:
                if not kept:
                    approx_chars = max(80, budget * 4)
                    return line[:approx_chars].rstrip()
                break
            kept.append(line)
            used += line_cost
        return "\n".join(kept).strip()

    def _render_task(self, goal: str, status: str, outcome: str | None) -> str:
        lines = [f"- Goal: {goal}", f"- Status: {status}"]
        if outcome:
            lines.append(f"- Outcome: {outcome}")
        return "\n".join(lines)

    def _render_events(self, events: list[AgentEvent]) -> str:
        return "\n".join(
            f"- [{event.role}/{event.event_type}] {event.content}"
            for event in events
            if event.content
        )

    def _render_checkpoints(self, checkpoints: list) -> str:
        lines: list[str] = []
        for checkpoint in checkpoints:
            lines.append(f"- {checkpoint.summary}")
            for item in checkpoint.pending_steps[:5]:
                lines.append(f"  pending: {item}")
            for item in checkpoint.blockers[:5]:
                lines.append(f"  blocker: {item}")
        return "\n".join(lines)

    def _render_decisions(self, checkpoints: list, events: list[AgentEvent]) -> str:
        lines: list[str] = []
        for checkpoint in checkpoints:
            lines.extend(f"- {item}" for item in checkpoint.decisions[:10])
        for event in events:
            if event.event_type == "decision" and event.content:
                lines.append(f"- {event.content}")
        return "\n".join(lines)

    def _render_artifacts(self, events: list[AgentEvent]) -> str:
        lines: list[str] = []
        for event in events:
            if event.event_type in {"artifact", "tool_call", "tool_result"}:
                if event.content:
                    lines.append(f"- [{event.event_type}] {event.content}")
                elif event.payload:
                    lines.append(f"- [{event.event_type}] {event.payload}")
        return "\n".join(lines)
