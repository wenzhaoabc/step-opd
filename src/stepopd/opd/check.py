import argparse
import json
import math
from pathlib import Path

import httpx

from stepopd.common import read_jsonl

from .episode import PROMPTS, make_environment


def check(environment, data_dir, episodes=1, retrieval_url="http://127.0.0.1:8000/retrieve"):
    records = read_jsonl(Path(data_dir) / "train.jsonl")
    if not records:
        raise ValueError("Missing/empty train.jsonl; run stepopd.opd.prepare first")
    if environment == "search_r1":
        response = httpx.post(retrieval_url, json={"queries": [records[0]["task"]["question"]],
                              "topk": 3, "return_scores": True}, timeout=30)
        response.raise_for_status()
        result = response.json()["result"]
        if len(result) != 1 or not result[0]:
            raise ValueError("Retriever readiness query produced no documents")
        for item in result[0]:
            if not isinstance(item["document"]["contents"], str) or not math.isfinite(item["score"]):
                raise ValueError("Retriever returned invalid document/score")
    reports = []
    for row in records[:episodes]:
        if row["environment"] != environment:
            raise ValueError("Manifest environment differs from --environment")
        env = make_environment(environment, row["task"], data_dir, retrieval_url)
        try:
            before = env.reset()
            if environment == "alfworld":
                response = f"<action>{env.actions[0]}</action>"
            elif environment == "scienceworld":
                response = "<action>look around</action>"
            else:
                response = f"<search>{row['task']['question']}</search>"
            after = env.step(response)
            if not before.text or not after.text or after.invalid_action:
                raise ValueError("Environment reset/step returned invalid observations")
            reports.append({"id": row["id"], "observation_chars": len(before.text),
                            "step_chars": len(after.text), "score": after.score})
        finally:
            env.close()
    return {"environment": environment, "checked": len(reports), "episodes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", required=True, choices=list(PROMPTS))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--retrieval-url", default="http://127.0.0.1:8000/retrieve")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    print(json.dumps(check(args.environment, args.data_dir, args.episodes, args.retrieval_url), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
