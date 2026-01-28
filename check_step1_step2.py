import os
from parser import parse_documents_from_file, parse_queries_from_file, parse_qrels_from_tsv
from preprocessing import load_stopwords_from_file, preprocess_documents, preprocess_queries
from indexing import build_inverted_index

DATASET_DIR = os.path.join("datasets", "scifact")
CORPUS_PATH = os.path.join(DATASET_DIR, "corpus.jsonl")
QUERIES_PATH = os.path.join(DATASET_DIR, "queries.jsonl")
QRELS_PATH = os.path.join(DATASET_DIR, "qrels", "test.tsv")

STOPWORDS_PATH = "stopwords.txt"  # make sure this file exists (1 word per line)


def main():
    # load
    docs = parse_documents_from_file(CORPUS_PATH)
    queries = parse_queries_from_file(QUERIES_PATH)
    qrels = parse_qrels_from_tsv(QRELS_PATH)

    # stopwords
    stop = load_stopwords_from_file(STOPWORDS_PATH)

    # preprocess
    docs = preprocess_documents(docs, stop)
    queries = preprocess_queries(queries, stop)

    # build index
    index, doc_lengths, doc_freq = build_inverted_index(docs)

    # ---- CHECKS / OUTPUTS (for README) ----
    vocab = list(index.keys())
    print("Docs:", len(docs))
    print("Queries (raw):", len(queries))
    print("Qrels queries:", len(qrels))
    print("Vocabulary size |V|:", len(vocab))

    # sample 100 tokens
    sample_100 = vocab[:100]
    print("\nSample 100 tokens:")
    print(sample_100)

    # postings sanity
    if vocab:
        term = vocab[0]
        posting = index[term]
        print(f"\nExample term: {term}")
        print("DF:", doc_freq[term])
        print("First 5 postings (doc_id -> tf):", list(posting.items())[:5])

    # show one preprocessed document sample
    d0 = docs[0]
    print("\nDoc sample:")
    print("DOCNO:", d0["DOCNO"])
    print("HEAD_TOKENS (first 20):", d0.get("HEAD_TOKENS", [])[:20])
    print("TEXT_TOKENS (first 20):", d0.get("TEXT_TOKENS", [])[:20])

    # show one preprocessed query sample
    q0 = queries[0]
    print("\nQuery sample:")
    print("num:", q0["num"])
    print("TOKENS:", q0.get("TOKENS", [])[:30])


if __name__ == "__main__":
    main()
