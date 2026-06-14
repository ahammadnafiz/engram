"""Smoke tests for Markdown Python examples."""

from __future__ import annotations

import ast
import re
from pathlib import Path

DOC_PATHS = [Path("README.md"), *sorted(Path("docs").glob("*.md"))]


def test_markdown_python_blocks_compile() -> None:
    failures: list[str] = []
    pattern = re.compile(r"```python\n(.*?)\n```", re.DOTALL)

    for path in DOC_PATHS:
        text = path.read_text(encoding="utf-8")
        for index, match in enumerate(pattern.finditer(text), start=1):
            code = match.group(1)
            try:
                compile(
                    code,
                    f"{path} python block {index}",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
            except SyntaxError as exc:
                failures.append(
                    f"{path}: python block {index}: {exc.msg} on line {exc.lineno}"
                )

    assert failures == []
