"""Input path discovery and task input assembly."""


from .utils import *


def infer_simulator_from_paths(args: argparse.Namespace, task_name: str) -> str:
    requested = (args.simulator or "auto").strip().lower()
    if requested != "auto":
        return requested

    return "rlbench"


def parse_path_entry_spec(spec: str, default_label: str | None = None) -> PathEntry:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return PathEntry(label.strip(), Path(path).expanduser())
    path = Path(spec).expanduser()
    return PathEntry(default_label or path.stem, path)


def build_task_paths(args: argparse.Namespace, task_name: str, grid_image: Path | None = None) -> TaskInputs:
    simulator = infer_simulator_from_paths(args, task_name)
    photo_dir = Path(args.photo_dir).expanduser()
    grid_image_path = as_path(args.grid_image) or grid_image or find_grid_image(task_name, photo_dir)
    output_json_path = (
        as_path(args.output_json)
        or grid_image_path.with_suffix(f".{args.provider}.multimodal_analysis.json")
    )

    gripper_sequence_path = (
        as_path(getattr(args, "gripper_sequence", None))
        or Path(getattr(args, "gripper_sequence_dir", DEFAULT_GRIPPER_SEQUENCE_DIR)).expanduser() / f"{task_name}.json"
    )

    ttm_context_path = as_path(args.ttm_context) or find_ttm_context(task_name)
    image_paths = [PathEntry("combined_grid", grid_image_path)]
    image_paths.extend(
        parse_path_entry_spec(spec, "extra_image")
        for spec in getattr(args, "image_path", [])
    )

    scene_context_paths = []
    if ttm_context_path is not None:
        scene_context_paths.append(PathEntry("scene_object_relationship_context", ttm_context_path))
    scene_context_paths.extend(
        parse_path_entry_spec(spec, "extra_scene_context")
        for spec in getattr(args, "scene_context_path", [])
    )

    return TaskInputs(
        task_name=task_name,
        simulator=simulator,
        image_paths=image_paths,
        scene_context_paths=scene_context_paths,
        output_paths=[PathEntry("multimodal_analysis_json", output_json_path)],
        gripper_sequence_path=gripper_sequence_path,
    )
