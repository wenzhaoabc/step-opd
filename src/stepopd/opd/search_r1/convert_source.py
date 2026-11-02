import argparse
import json
from pathlib import Path


def _load_rows(path: Path, data_source: str | None = None) -> tuple[list[dict], int]:
    import pyarrow.parquet as pq

    columns = ["question", "golden_answers"]
    if data_source is not None:
        if "data_source" not in pq.read_schema(path).names:
            raise ValueError(f"input parquet lacks a data_source column, cannot filter by subset: {path}")
        columns.append("data_source")
    table = pq.read_table(path, columns=columns)
    questions = table.column("question").to_pylist()
    answers = table.column("golden_answers").to_pylist()
    sources = table.column("data_source").to_pylist() if data_source else [None] * len(questions)
    rows, filtered_out = [], 0
    for question, golden_answers, source in zip(questions, answers, sources):
        if data_source is not None and source != data_source:
            filtered_out += 1
            continue
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"question must be a non-empty string: {path}")
        if (
            not isinstance(golden_answers, list)
            or not golden_answers
            or not all(isinstance(x, str) and x.strip() for x in golden_answers)
        ):
            raise ValueError(f"golden_answers must be a list of non-empty strings: {path}")
        rows.append(
            {
                "question": question.strip(),
                "golden_answers": [item.strip() for item in golden_answers],
            }
        )
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows, filtered_out


def convert(source_dir: Path, output_dir: Path, data_source: str | None = None) -> dict:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("parquet source dir and output dir must differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"source_dir": str(source_dir), "output_dir": str(output_dir), "splits": {}}
    if data_source is not None:
        stats["data_source"] = data_source
    seen = set()
    test_questions = None
    for split in ("test", "train"):
        source = source_dir / f"{split}.parquet"
        if not source.is_file():
            raise FileNotFoundError(source)
        rows, filtered_out = _load_rows(source, data_source)
        if split == "test":
            test_questions = {row["question"] for row in rows}
        kept, dropped_within, dropped_overlap = [], 0, 0
        for row in rows:
            if split == "train" and row["question"] in test_questions:
                dropped_overlap += 1
                continue
            if row["question"] in seen:
                dropped_within += 1
                continue
            seen.add(row["question"])
            kept.append(row)
        target = output_dir / f"{split}.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in kept:
                stream.write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
            stream.flush()
        temporary.replace(target)
        stats["splits"][split] = {
            "input_rows": len(rows),
            "output_rows": len(kept),
            "dropped_duplicate_within_split": dropped_within,
            "dropped_duplicate_across_splits": dropped_overlap,
        }
        if data_source is not None:
            stats["splits"][split]["filtered_out_other_data_source"] = filtered_out
    from stepopd.common import write_json

    write_json(output_dir / "source_manifest.json", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="download directory containing train.parquet/test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="canonical source directory for train.jsonl/test.jsonl output",
    )
    parser.add_argument(
        "--data-source",
        help="keep only this data_source subset (e.g. hotpotqa, the MultiQA multi-turn retrieval subset)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            convert(args.source_dir, args.output_dir, args.data_source),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
