import re
import string

import httpx

from ..episode import State, parse_command


def normalize_answer(value):
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


class SearchR1:
    max_steps = 5

    def __init__(self, task, retrieval_url):
        self.question = task["question"]
        self.answers = task["golden_answers"]
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a nonempty string")
        if (
            not isinstance(self.answers, list)
            or not self.answers
            or not all(isinstance(x, str) and x.strip() for x in self.answers)
        ):
            raise ValueError(
                "golden_answers must be a nonempty list of nonempty strings"
            )
        self.retrieval_url = retrieval_url
        self.searches = 0
        self.client = httpx.Client(timeout=30.0)

    def reset(self):
        self.searches = 0
        return State("Question: " + self.question)

    def step(self, response):
        try:
            tag, value = parse_command(response, {"search", "answer"})
        except ValueError as error:
            return State(str(error), invalid_action=True)
        if tag == "answer":
            success = normalize_answer(value) in {
                normalize_answer(answer) for answer in self.answers
            }
            return State("Episode finished.", float(success), True, success, "answer")
        if self.searches == 4:
            return State(
                "The four-search limit was reached. Provide the final answer.",
                invalid_action=True,
            )
        result = self.client.post(
            self.retrieval_url,
            json={"queries": [value], "topk": 3, "return_scores": True},
        )
        result.raise_for_status()
        documents = result.json()["result"]
        if len(documents) != 1:
            raise ValueError(
                "Retriever must return exactly one result list for one query"
            )
        text = "\n\n".join(
            f"Document {i + 1}: {item['document']['contents']}"
            for i, item in enumerate(documents[0])
        )
        self.searches += 1
        return State(
            f"<information>\n{text}\n</information>\nSearches remaining: {4 - self.searches}."
        )

    def close(self):
        self.client.close()
