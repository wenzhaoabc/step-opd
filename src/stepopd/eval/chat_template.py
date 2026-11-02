import json
from pathlib import Path

NON_THINKING_CONDITION = "enable_thinking is defined and enable_thinking is false"
PATCHED_NON_THINKING_CONDITION = f"(true or ({NON_THINKING_CONDITION}))"


def patch_non_thinking(hf_dir: Path) -> None:
    config_path = hf_dir / "tokenizer_config.json"
    template_path = hf_dir / "chat_template.jinja"
    if template_path.is_file():
        template = template_path.read_text()
    else:
        template_path = config_path
        config = json.loads(config_path.read_text())
        template = config["chat_template"]
    if PATCHED_NON_THINKING_CONDITION in template:
        return
    if NON_THINKING_CONDITION not in template:
        raise ValueError(
            f"chat_template is missing the non-thinking condition ({NON_THINKING_CONDITION}); "
            "the template version may have changed, patch semantics need manual review"
        )
    patched = template.replace(NON_THINKING_CONDITION, PATCHED_NON_THINKING_CONDITION)
    if template_path == config_path:
        config["chat_template"] = patched
        patched = json.dumps(config, ensure_ascii=False, indent=2)
    temporary = template_path.with_suffix(template_path.suffix + ".tmp")
    temporary.write_text(patched)
    temporary.replace(template_path)
