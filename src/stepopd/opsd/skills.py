import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[3]
ENVIRONMENTS = ("alfworld", "scienceworld", "search_r1")
SEARCH_CATEGORIES = {
    "hotpotqa": "multi_hop_reasoning",
    "2wikimultihopqa": "multi_hop_reasoning",
    "musique": "multi_hop_reasoning",
    "nq": "direct_retrieval",
    "natural_questions": "direct_retrieval",
    "triviaqa": "entity_attribute_lookup",
    "popqa": "entity_attribute_lookup",
}


@dataclass(frozen=True)
class SkillDocument:
    name: str
    path: str
    text: str
    sha256: str


@dataclass(frozen=True)
class SkillSelection:
    category: str
    documents: tuple[SkillDocument, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(
            f"## {item.name}\n{item.text.strip()}" for item in self.documents
        )

    def evidence(self) -> dict:
        return {
            "category": self.category,
            "files": [
                {"path": item.path, "sha256": item.sha256} for item in self.documents
            ],
        }


class SkillBank:

    def __init__(self, root, environment, default_category=None):
        if environment not in ENVIRONMENTS:
            raise ValueError(f"Unsupported SKILL environment: {environment}")
        root = Path(root).expanduser()
        self.root = (root if root.is_absolute() else CODE_ROOT / root).resolve()
        self.environment = environment
        directory = self.root / environment
        if environment == "search_r1" and not directory.exists():
            directory = self.root / "search"
        directory = directory.resolve()
        mapping_path = directory / "skill_mapping.json"
        raw_mapping = mapping_path.read_bytes()
        mapping = json.loads(raw_mapping)
        files = mapping.get("skill_files", {})
        self.task_to_skill = mapping.get("task_to_skill", {})
        if not isinstance(files, dict) or "general_skills" not in files:
            raise ValueError(f"{mapping_path}: skill_files must contain general_skills")
        if not isinstance(self.task_to_skill, dict):
            raise ValueError(f"{mapping_path}: task_to_skill must be an object")
        self.documents = {}
        for name, relative in files.items():
            if not isinstance(relative, str):
                raise ValueError(f"Invalid SKILL filename for {name}")
            path = (directory / relative).resolve()
            if not path.is_relative_to(directory) or path.suffix.lower() != ".md":
                raise ValueError(
                    f"SKILL must be a Markdown file inside {directory}: {relative}"
                )
            raw = path.read_bytes()
            text = raw.decode("utf-8")
            if not text.strip():
                raise ValueError(f"Empty SKILL: {path}")
            self.documents[name] = SkillDocument(
                name, relative, text, hashlib.sha256(raw).hexdigest()
            )
        for task, name in self.task_to_skill.items():
            if (
                not isinstance(task, str)
                or not isinstance(name, str)
                or name not in self.documents
            ):
                raise ValueError(f"Unknown SKILL mapping: {task} -> {name}")
        self.default_category = default_category
        if default_category is not None:
            self._resolve_category(default_category)
        self.manifest = {
            "environment": environment,
            "mapping_sha256": hashlib.sha256(raw_mapping).hexdigest(),
            "default_category": default_category,
            "documents": [
                {"name": doc.name, "path": doc.path, "sha256": doc.sha256}
                for doc in sorted(self.documents.values(), key=lambda item: item.name)
            ],
        }
        self.sha256 = hashlib.sha256(
            json.dumps(self.manifest, sort_keys=True).encode()
        ).hexdigest()

    def _resolve_category(self, category):
        name = self.task_to_skill.get(category, category)
        if name not in self.documents:
            raise ValueError(f"Unknown {self.environment} SKILL category: {category}")
        return name

    def select(self, task: dict) -> SkillSelection:
        category = task.get("skill_category")
        if category is None and self.environment == "alfworld":
            task_type = task.get("task_type", "")
            parts = [
                task_type,
                *str(task.get("game_file", task.get("gamefile", ""))).split("/"),
            ]
            for candidate in sorted(self.task_to_skill, key=len, reverse=True):
                if any(
                    part == candidate
                    or part.startswith(candidate + "-")
                    or part.startswith(candidate + "_")
                    for part in parts
                ):
                    category = candidate
                    break
        elif category is None and self.environment == "scienceworld":
            category = self.task_to_skill.get(task.get("task_name", ""))
        elif category is None and self.environment == "search_r1":
            source = task.get("source_dataset", task.get("data_source"))
            if source:
                category = SEARCH_CATEGORIES.get(str(source).lower(), "general_skills")
        if category is None:
            category = self.default_category or "general_skills"
        name = self._resolve_category(category)
        names = ["general_skills"] + ([name] if name != "general_skills" else [])
        return SkillSelection(
            str(category), tuple(self.documents[key] for key in names)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="resources/opsd_skills")
    parser.add_argument("--environment", required=True, choices=ENVIRONMENTS)
    parser.add_argument("--category")
    parser.add_argument("--show-text", action="store_true")
    args = parser.parse_args()
    bank = SkillBank(args.root, args.environment, args.category)
    print(
        json.dumps(
            {**bank.manifest, "root": str(bank.root), "sha256": bank.sha256}, indent=2
        )
    )
    if args.show_text:
        print(bank.select({}).text)


if __name__ == "__main__":
    main()
