#!/usr/bin/env python3
"""Public workflow entry point. Run with --help for usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aha_publish import paths
from aha_publish.commands import entry, parser, require, run, selected_tasks

def main():
    p = parser("Step 3: generate behavior-tree conditions from task descriptions.", tasks=True)
    p.add_argument("--provider", choices=["openai"], default="openai", help="Provider used to generate the BT.")
    p.add_argument("--description-provider", choices=["openai"], default="openai")
    p.add_argument("--model", help="Override the BT generation and review model.")
    p.add_argument("--no-reviewer", action="store_true")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true", help="Generate and save without opening the web interface. Multiple tasks always use this mode.")
    mode.add_argument("--no-auto-generate", action="store_true", help="Open the saved BT in the web interface without regenerating it. Requires one task.")
    args = p.parse_args()
    tasks = selected_tasks(args)
    if args.no_auto_generate and (args.all or len(tasks) != 1):
        p.error("--no-auto-generate requires exactly one --task and cannot be used with --all")
    generate_only = args.generate_only or len(tasks) > 1 or args.all
    for task in tasks:
        description = paths.DESCRIPTION_DIR / f"{task}_ALL_WAYPOINTS_COMBINED.{args.description_provider}.multimodal_analysis.json"
        require(description, args.dry_run)
        options = ["--task-context", description, "--provider", args.provider]
        if args.no_auto_generate:
            require(paths.BT_DIR / f"{task}.bt_conditions.json", args.dry_run)
            options += ["--no-auto-generate"]
        else:
            options += ["--regenerate"]
        if generate_only:
            options += ["--generate-only"]
        if args.model:
            options += ["--model", args.model]
        if args.no_reviewer:
            options += ["--no-reviewer"]
        run("behavior_trees.cli", options, args.dry_run)
        if generate_only:
            require(paths.BT_DIR / f"{task}.bt_conditions.json", args.dry_run)


if __name__ == "__main__":
    entry(main)
