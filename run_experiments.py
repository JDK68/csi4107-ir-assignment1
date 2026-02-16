import os
import subprocess
import sys

from parser import (
    parse_documents_from_file,
    parse_queries_from_file,
    parse_qrels_from_tsv,
    filter_queries_by_qrels_and_odd,
)
from preprocessing import load_stopwords_from_file, preprocess_documents, preprocess_queries
from indexing import build_inverted_index
from retrieval_and_ranking import compute_similarity_scores

DATASET_DIR = os.path.join("datasets", "scifact")
if not os.path.isdir(DATASET_DIR) or not os.path.isfile(os.path.join(DATASET_DIR, "corpus.jsonl")):
    DATASET_DIR = "scifact"
CORPUS_PATH = os.path.join(DATASET_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(DATASET_DIR, "queries.jsonl")
QRELS_PATH = os.path.join(DATASET_DIR, "qrels", "test.tsv")
STOPWORDS_PATH = "stopwords.txt"

RUN_TAG = "csi4107_vsm"
TOP_K = 100

def write_trec_results(results_path: str, run_name: str, query_ranked: list):
    """
    query_ranked: list of (query_id, list of (doc_id, score)) in ascending query_id order.
    Format: query_id Q0 doc_id rank score tag
    """
    with open(results_path, "w", encoding="utf-8") as f:
        for query_id, ranked_docs in query_ranked:
            for rank, (doc_id, score) in enumerate(ranked_docs[:TOP_K], start=1):
                line = f"{query_id} Q0 {doc_id} {rank} {score:.4f} {run_name}\n"
                f.write(line)


def write_qrels_trec_format(qrels: dict, out_path: str):
    """Write qrels in trec_eval format: query_id 0 docno relevance"""
    with open(out_path, "w", encoding="utf-8") as f:
        for qid in sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            for docid, rel in qrels[qid].items():
                f.write(f"{qid} 0 {docid} {rel}\n")


def compute_map_python(qrels: dict, query_ranked: list) -> float:
    """Compute Mean Average Precision from qrels and ranked results (list of (query_id, [(doc_id, score), ...]))."""
    aps = []
    qrels_by_q = qrels
    for query_id, ranked in query_ranked:
        qid_str = str(query_id)
        rel_docs = set(qrels_by_q.get(qid_str, {}).keys())
        if not rel_docs:
            continue
        num_rel = len(rel_docs)
        prec_sum = 0.0
        num_ret_rel = 0
        for k, (doc_id, _) in enumerate(ranked[:TOP_K], 1):
            if doc_id in rel_docs:
                num_ret_rel += 1
                prec_sum += num_ret_rel / k
        ap = prec_sum / num_rel if num_rel else 0.0
        aps.append(ap)
    return sum(aps) / len(aps) if aps else 0.0


def run_trec_eval(qrels_path: str, results_path: str) -> dict:
    """Run trec_eval and return dict of metric -> value. Returns empty dict if trec_eval not found."""
    if not os.path.isfile(qrels_path) or not os.path.isfile(results_path):
        return {}
    cmd = ["trec_eval", "-m", "map", "-m", "recip_rank", "-m", "P.10", qrels_path, results_path]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            return {"error": out.stderr or out.stdout}
        metrics = {}
        for line in out.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                metrics[parts[0]] = float(parts[2])
        return metrics
    except FileNotFoundError:
        return {"error": "trec_eval not found in PATH"}
    except subprocess.TimeoutExpired:
        return {"error": "trec_eval timed out"}


def main():
    if not os.path.isfile(CORPUS_PATH):
        print(f"Corpus not found at {CORPUS_PATH}. Place Scifact corpus.jsonl there (or in scifact/).", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(STOPWORDS_PATH):
        print(f"Stopwords file not found: {STOPWORDS_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading corpus, queries, qrels...")
    docs = parse_documents_from_file(CORPUS_PATH)
    queries_all = parse_queries_from_file(QUERIES_PATH)
    qrels = parse_qrels_from_tsv(QRELS_PATH)
    test_queries = filter_queries_by_qrels_and_odd(queries_all, qrels)
    test_queries.sort(key=lambda q: int(q["num"]) if str(q["num"]).isdigit() else 0)

    print("Preprocessing...")
    stop = load_stopwords_from_file(STOPWORDS_PATH)
    preprocess_documents(docs, stop)
    preprocess_queries(test_queries, stop)

    # Build doc_id -> doc for retrieval
    doc_list = docs
    doc_ids = set(str(d["DOCNO"]) for d in docs)

    # Qrels in trec_eval format
    qrels_trec_path = os.path.join(DATASET_DIR, "qrels", "test_trec_eval.qrels")
    os.makedirs(os.path.dirname(qrels_trec_path), exist_ok=True)
    write_qrels_trec_format(qrels, qrels_trec_path)
    print(f"Wrote qrels for trec_eval: {qrels_trec_path}")

    runs = [
        ("title_only", "HEAD_TOKENS", "Results_title_only.txt"),
        ("title_and_text", "TOKENS", "Results_title_and_text.txt"),
    ]

    all_results = []
    map_scores = {}

    for run_name, token_field, out_file in runs:
        print(f"\n--- Run: {run_name} (index field = {token_field}) ---")
        index, doc_lengths, doc_freq = build_inverted_index(docs, token_field=token_field)

        query_ranked = []
        for i, q in enumerate(test_queries):
            ranked = compute_similarity_scores(q, index, doc_list)
            query_ranked.append((q["num"], ranked))
            if (i + 1) % 50 == 0:
                print(f"  Queries processed: {i + 1}/{len(test_queries)}")

        write_trec_results(out_file, RUN_TAG, query_ranked)
        print(f"  Wrote {out_file}")

        metrics = run_trec_eval(qrels_trec_path, out_file)
        if "error" in metrics:
            map_py = compute_map_python(qrels, query_ranked)
            metrics["map"] = map_py
            print(f"  trec_eval: {metrics['error']}")
            print(f"  MAP (computed in Python) = {map_py:.4f}")
        else:
            print(f"  MAP (trec_eval) = {metrics.get('map', 0):.4f}")
        map_scores[run_name] = metrics.get("map")
        all_results.append((run_name, query_ranked, metrics))

    # Best run by MAP -> write to "Results"
    best_run_name = None
    best_map = -1.0
    for run_name, _, metrics in all_results:
        m = metrics.get("map")
        if m is not None and m > best_map:
            best_map = m
            best_run_name = run_name

    if best_run_name is not None:
        for rname, qranked, _ in all_results:
            if rname == best_run_name:
                write_trec_results("Results", RUN_TAG, qranked)
                print(f"\nBest run: {best_run_name} (MAP = {best_map:.4f}). Wrote 'results'.")
                break
    else:
        for rname, qranked, _ in all_results:
            if rname == "title_and_text":
                write_trec_results("Results", RUN_TAG, qranked)
                print("\nWrote 'results' (title_and_text). Run trec_eval manually for MAP.")

    # Print first 10 answers for first 2 queries
    print("\n--- First 10 results for first 2 queries (best run) ---")
    for rname, qranked, _ in all_results:
        if rname == (best_run_name or "title_and_text"):
            for qid, ranked in qranked[:2]:
                print(f"Query {qid}:")
                for r, (doc_id, score) in enumerate(ranked[:10], 1):
                    print(f"  {r}. {doc_id} {score:.4f}")
            break

    return map_scores


if __name__ == "__main__":
    main()
