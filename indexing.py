from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# Inverted index type:
# index[term][doc_id] = tf
InvertedIndex = Dict[str, Dict[str, int]]


def build_inverted_index(documents: List[dict]) -> Tuple[InvertedIndex, Dict[str, int], Dict[str, int]]:
    """
    Step 2: Build inverted index from preprocessed tokens.

    Returns:
      - inverted_index: term -> {doc_id: tf}
      - doc_lengths: doc_id -> number of tokens (for BM25 later)
      - doc_freq: term -> df (number of docs containing term)
    """
    inverted_index: InvertedIndex = defaultdict(dict)
    doc_lengths: Dict[str, int] = {}
    doc_freq: Dict[str, int] = {}

    for doc in documents:
        doc_id = str(doc["DOCNO"])
        tokens = doc.get("TOKENS", [])
        doc_lengths[doc_id] = len(tokens)

        tf = Counter(tokens)
        for term, freq in tf.items():
            inverted_index[term][doc_id] = freq

    # df
    for term, posting in inverted_index.items():
        doc_freq[term] = len(posting)

    return dict(inverted_index), doc_lengths, doc_freq
