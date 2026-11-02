import argparse
import math
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field


class Query(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=32)
    topk: int = Field(default=3, ge=1, le=10)
    return_scores: Literal[True] = True


def build_app(index_path, corpus_path):
    from datasets import load_dataset
    from pyserini.pyclass import autoclass

    index_path, corpus_path = Path(index_path), Path(corpus_path)
    if not index_path.is_dir() or not corpus_path.is_file():
        raise FileNotFoundError("BM25 requires a mounted Lucene index directory and matching corpus JSONL")
    searcher = autoclass("io.anserini.search.SimpleSearcher")(str(index_path))
    searcher.set_bm25(0.9, 0.4)
    corpus = load_dataset("json", data_files=str(corpus_path), split="train",
                          cache_dir=str(corpus_path.parent / ".arrow_cache"))
    if not len(corpus) or "id" not in corpus.column_names or "contents" not in corpus.column_names:
        raise ValueError("Corpus requires nonempty id/contents columns")
    if searcher.get_total_num_docs() != len(corpus):
        raise ValueError("Lucene index and corpus document counts differ")
    app = FastAPI(title="stepopd CPU BM25")

    @app.get("/health")
    def health():
        return {"ready": True, "documents": len(corpus), "retriever": "bm25"}

    @app.post("/retrieve")
    def retrieve(request: Query):
        result = []
        for query in request.queries:
            if not query.strip():
                raise ValueError("Search query must be nonempty")
            documents = []
            for hit in searcher.search(query, request.topk):
                item = corpus[int(hit.docid)]
                if str(item["id"]) != hit.docid or not isinstance(item["contents"], str):
                    raise ValueError(f"Index/corpus mismatch for docid={hit.docid}")
                score = float(hit.score)
                if not math.isfinite(score):
                    raise ValueError("Retriever returned a nonfinite score")
                documents.append({"document": {"id": str(item["id"]), "contents": item["contents"]}, "score": score})
            result.append(documents)
        return {"result": result}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    os.environ["JAVA_HOME"] = os.environ["BM25_JAVA_HOME"]
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    print("Loading CPU BM25 index and corpus; first Arrow conversion may take several minutes.", flush=True)
    app = build_app(args.index_path, args.corpus_path)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
