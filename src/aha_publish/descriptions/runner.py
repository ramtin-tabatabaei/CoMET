"""Task execution, output writing, usage reporting, and task discovery."""



from .providers import *

def analyze_task(task_paths: TaskInputs, args: argparse.Namespace, client) -> tuple[dict[str, Any], Usage]:
    if args.provider != "openai":
        raise ValueError(f"Unsupported provider '{args.provider}'. Only OpenAI is supported.")
    waypoints = discover_waypoints(task_paths, args)
    text_payload = build_text_payload(task_paths, args, waypoints)

    if args.write_prompt:
        prompt_path = Path(args.write_prompt).expanduser()
        if args.all:
            prompt_path = prompt_path.with_name(
                f"{prompt_path.stem}.{task_paths.task_name}{prompt_path.suffix}"
            )
        prompt_path.write_text(text_payload)

    if args.dry_run:
        return {
            "task": task_paths.task_name,
            "dry_run": True,
            "expected_waypoint_ids": waypoints,
            "prompt_chars": len(text_payload),
        }, Usage()

    return analyze_with_openai(task_paths, args, text_payload, client)


def print_analysis_summary(task_paths: TaskInputs, analysis: dict[str, Any]) -> None:
    print(f"\nTask: {task_paths.task_name}")
    print(f"Output: {task_paths.output_json_path}")
    description = analysis.get("overall_description")
    if description:
        print(f"Overall: {description}")
    waypoints = analysis.get("waypoints", [])
    print(f"Waypoint entries: {len(waypoints)}")


def model_name_for(args: argparse.Namespace) -> str:
    return args.openai_model


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "n/a (set --input-price-per-1m and --output-price-per-1m)"
    return f"${cost:.4f}"


def usage_record(args: argparse.Namespace, usage: Usage, label: str) -> dict[str, Any]:
    cost = usage.cost(args.input_price_per_1m, args.output_price_per_1m)
    return {
        "label": label,
        "provider": args.provider,
        "model": model_name_for(args),
        "calls": usage.calls,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
        "input_price_per_1m": args.input_price_per_1m,
        "output_price_per_1m": args.output_price_per_1m,
        "cost_usd": cost,
    }


def print_usage_summary(args: argparse.Namespace, usage: Usage, label: str) -> None:
    cost = usage.cost(args.input_price_per_1m, args.output_price_per_1m)
    print(
        f"{label}: {usage.input_tokens} input + {usage.output_tokens} output "
        f"= {usage.total_tokens} tokens over {usage.calls} call(s) "
        f"on {model_name_for(args)}; cost {format_cost(cost)}"
    )


def run_one(task_name: str, args: argparse.Namespace, client, grid_image: Path | None = None) -> tuple[Path, Usage]:
    task_paths = build_task_paths(args, task_name, grid_image)
    print(f"\nAnalyzing {task_paths.task_name}")
    print(f"Grid image: {task_paths.grid_image_path}")

    analysis, usage = analyze_task(task_paths, args, client)
    task_paths.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    task_paths.output_json_path.write_text(json.dumps(analysis, indent=2))

    print_analysis_summary(task_paths, analysis)

    if not args.dry_run:
        usage_path = task_paths.output_json_path.with_suffix(".usage.json")
        usage_path.write_text(
            json.dumps(usage_record(args, usage, task_paths.task_name), indent=2)
        )
        print_usage_summary(args, usage, f"Cost [{task_paths.task_name}]")

    return task_paths.output_json_path, usage


def discover_generator_tasks(photo_dir: Path, provider: str) -> list[tuple[str, bool, bool]]:
    """Return [(task_name, has_grid_image, already_generated)] per task, sorted.

    Only tasks that actually have a grid image in photo_dir are listed — a task
    without one has nothing to analyze, so it is not offered. already_generated
    is True when the analysis JSON this run would write already exists
    (re-running overwrites it).
    """
    tasks: dict[str, bool] = {}
    if photo_dir.exists():
        for image in find_all_grid_images(photo_dir):
            task_name = extract_task_from_grid_image(image)
            analysis = image.with_suffix(f".{provider}.multimodal_analysis.json")
            tasks[task_name] = analysis.exists()
    return sorted((name, True, generated) for name, generated in tasks.items())


def prompt_for_task(photo_dir: Path, provider: str) -> list[str] | None:
    """Show a numbered menu and return tasks selected by numbers or inclusive ranges.

    Returns None when there is nothing to choose from or stdin is not a
    terminal, so callers can fall back to the default. Exits cleanly if quit.
    """
    tasks = discover_generator_tasks(photo_dir, provider)
    if not tasks or not sys.stdin.isatty():
        return None

    generated = sum(1 for _, _, done in tasks if done)
    print(f"\nAvailable tasks ({len(tasks)} with a grid image, {generated} already generated):")
    selectable: dict[int, str] = {}
    for index, (name, _has_image, done) in enumerate(tasks, start=1):
        marker = "  — already generated (will overwrite)" if done else ""
        selectable[index] = name
        print(f"  {index:>3}. {name}{marker}")

    while True:
        try:
            choice = input(
                f"\nSelect task(s) to generate [1-{len(tasks)}] "
                "(e.g. 31-40 or 1,3-5; q to quit): "
            ).strip()
        except EOFError:
            return None
        if choice.lower() in ("q", "quit", "exit"):
            print("No task selected. Exiting.")
            sys.exit(0)
        selected: dict[int, None] = {}
        try:
            for part in choice.split(","):
                bounds = part.strip().split("-")
                if len(bounds) not in (1, 2) or not all(b.strip().isdecimal() for b in bounds):
                    raise ValueError
                start, end = int(bounds[0]), int(bounds[-1])
                if not 1 <= start <= end <= len(tasks):
                    raise ValueError
                selected.update((index, None) for index in range(start, end + 1))
        except ValueError:
            print(f"Please enter numbers or ascending ranges between 1 and {len(tasks)} (e.g. 1,3-5).")
            continue
        return [selectable[index] for index in selected]
