"""Task-context and path I/O helpers for BT maker."""

from aha_publish import paths

from .prompts import *

def parse_json_response(raw: str) -> dict:
    """Strip accidental markdown fences and parse JSON."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = "\n".join(
            line for line in clean.splitlines()
            if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        preview = clean[:900]
        if len(clean) > len(preview):
            preview += "..."
        raise ValueError(
            f"Model returned invalid JSON: {e}. Response preview: {preview}"
        ) from e

def load_json_file(path: Path) -> dict:
    """Read the current JSON so edits and caller mutations cannot go stale."""
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_task_context(path: Path) -> str:
    """Load task context verbatim, preserving explicit gripper/holding states."""
    return json.dumps(load_json_file(path), indent=2)

def load_task_context_object(path: Path) -> dict:
    return load_json_file(path)

def task_name_from_context_path(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    for marker in (
        "_ALL_WAYPOINTS_COMBINED",
        ".openai.multimodal_analysis",
        ".openai.analysis",
        ".multimodal_analysis",
        ".analysis",
    ):
        if marker in name:
            name = name.split(marker)[0]
    return name or path.stem

def task_name_from_context(path: Path, task_context: dict | None = None) -> str:
    task_context = task_context or {}
    for key in ("task", "task_name", "name"):
        value = str(task_context.get(key) or "").strip()
        if value:
            return value
    return task_name_from_context_path(path)

def default_review_output_path(
    task_context_path: Path,
    task_name: str | None = None,
) -> Path:
    task_name = task_name or task_name_from_context_path(task_context_path)
    return PREPARED_BTS_DIR / f"{task_name}.bt_conditions.json"

def resolve_project_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    candidates = [
        Path.cwd() / path,
        PROJECT_ROOT / path,
        AHA_SCRIPTS_DIR / path,
        SCRIPT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (PROJECT_ROOT / path).resolve()

def project_relative_str(path: Path | str) -> str:
    """Serialize a path portably: repo-relative when inside the repo, else absolute.

    Keeps saved *.bt_conditions.json files free of machine-specific absolute
    paths so they resolve on any checkout via resolve_project_path().
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(p)

def ensure_input_files(args: argparse.Namespace) -> None:
    if not args.task_context.exists():
        raise FileNotFoundError(f"Task context file not found: {args.task_context}")

def load_design_html() -> str:
    return (SCRIPT_DIR / "design.html").read_text()
