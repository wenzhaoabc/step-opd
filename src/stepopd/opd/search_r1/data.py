import hashlib
import json
import random


def prepare_records(source_dir, val_size):
    if source_dir is None:
        raise ValueError(
            "Search-R1 requires --source-dir containing train.jsonl and test.jsonl"
        )
    seen = set()
    for source_split in ["train", "test"]:
        path = source_dir / f"{source_split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        if not rows:
            raise ValueError(f"Empty Search-R1 input: {path}")
        if source_split == "train" and len(rows) <= val_size:
            raise ValueError(
                f"Need more than {val_size} training questions; adjust --val-size for a fixture"
            )
        random.Random(42).shuffle(rows)
        for i, row in enumerate(rows):
            question, answers = row["question"], row["golden_answers"]
            if not isinstance(question, str) or not question.strip():
                raise ValueError("question must be a nonempty string")
            if (
                not isinstance(answers, list)
                or not answers
                or not all(isinstance(x, str) and x.strip() for x in answers)
            ):
                raise ValueError(
                    "golden_answers must be a nonempty list of nonempty strings"
                )
            key = hashlib.sha256(question.strip().encode()).hexdigest()
            if key in seen:
                raise ValueError(
                    f"Duplicate question across input rows/splits: {question}"
                )
            seen.add(key)
            split = (
                "val" if source_split == "train" and i < val_size else source_split
            )
            yield (
                split,
                {
                    "id": "search_r1:" + key,
                    "environment": "search_r1",
                    "split": split,
                    "task": {
                        "question": question.strip(),
                        "golden_answers": answers,
                    },
                },
            )
