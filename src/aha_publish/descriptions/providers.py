"""Provider client calls for multimodal task description analysis."""

from aha_publish import paths

from .context_blocks import *

def analyze_with_openai(task_paths: TaskInputs, args: argparse.Namespace, text_payload: str, client) -> tuple[dict[str, Any], Usage]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": text_payload,
        }
    ]
    if args.include_grid_image:
        for image_entry in task_paths.image_paths:
            image_data, media_type = load_image_as_base64(image_entry.path)
            content.append({
                "type": "input_image",
                "image_url": f"data:{media_type};base64,{image_data}",
            })

    response = client.responses.create(
        model=args.openai_model,
        max_output_tokens=args.max_output_tokens,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )
    return parse_json_response(response.output_text), usage_from_openai(response)


def _load_openai_key() -> str | None:
    return os.environ.get("OPENAI_API_KEY")


def build_client(provider: str):
    if provider != "openai":
        raise ValueError(f"Unsupported provider '{provider}'. Only OpenAI is supported.")
    import openai
    api_key = _load_openai_key()
    if not api_key:
        raise RuntimeError(
            "OpenAI API key not found. Set OPENAI_API_KEY."
        )
    return openai.OpenAI(api_key=api_key)
