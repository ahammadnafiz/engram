from __future__ import annotations

import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "docker-setup.sh"


def copy_script(tmp_path: Path) -> Path:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    target = scripts_dir / "docker-setup.sh"
    shutil.copy2(SCRIPT, target)
    return target


def source_and_run(script: Path, command: str) -> None:
    subprocess.run(
        ["bash", "-c", f"source {shlex.quote(str(script))}; {command}"],
        cwd=script.parents[1],
        check=True,
        text=True,
        capture_output=True,
    )


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def test_docker_setup_preserves_existing_env_file(tmp_path: Path) -> None:
    script = copy_script(tmp_path)
    env_file = tmp_path / ".env"
    original = textwrap.dedent(
        """\
        POSTGRES_PORT=5444
        POSTGRES_USER=custom
        POSTGRES_PASSWORD=keepme
        POSTGRES_DB=customdb
        ENGRAM_DATABASE_URL=postgresql://custom:keepme@localhost:5444/customdb
        ENGRAM_OPENAI_API_KEY=sk-test
        ENGRAM_EMBEDDING_MODEL=text-embedding-3-small
        """
    )
    env_file.write_text(original)

    source_and_run(script, "create_or_update_env_file 5555 false")

    assert env_file.read_text() == original


def test_docker_setup_appends_missing_docker_defaults(tmp_path: Path) -> None:
    script = copy_script(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "ENGRAM_OPENAI_API_KEY=sk-test\nENGRAM_EMBEDDING_MODEL=text-embedding-3-small\n"
    )

    source_and_run(script, "create_or_update_env_file 5555 false")

    values = parse_env(env_file)
    assert values["ENGRAM_OPENAI_API_KEY"] == "sk-test"
    assert values["ENGRAM_EMBEDDING_MODEL"] == "text-embedding-3-small"
    assert values["POSTGRES_PORT"] == "5555"
    assert values["POSTGRES_USER"] == "engram"
    assert values["POSTGRES_DB"] == "engram"
    assert values["POSTGRES_PASSWORD"]
    assert (
        values["ENGRAM_DATABASE_URL"]
        == f"postgresql://engram:{values['POSTGRES_PASSWORD']}@localhost:5555/engram"
    )


def test_docker_setup_explicit_port_updates_managed_url_only(tmp_path: Path) -> None:
    script = copy_script(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "POSTGRES_PORT=5444\n"
        "POSTGRES_USER=engram\n"
        "POSTGRES_PASSWORD=keepme\n"
        "POSTGRES_DB=engram\n"
        "ENGRAM_DATABASE_URL=postgresql://engram:keepme@localhost:5444/engram\n"
        "ENGRAM_OPENAI_API_KEY=sk-test\n"
    )

    source_and_run(script, "create_or_update_env_file 5555 true")

    values = parse_env(env_file)
    assert values["POSTGRES_PORT"] == "5555"
    assert values["POSTGRES_PASSWORD"] == "keepme"
    assert values["ENGRAM_OPENAI_API_KEY"] == "sk-test"
    assert (
        values["ENGRAM_DATABASE_URL"]
        == "postgresql://engram:keepme@localhost:5555/engram"
    )
