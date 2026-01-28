from __future__ import annotations

import re
from typing import Iterable, List, Set, Dict

_WORD_RE = re.compile(r"[a-zA-Z]+")


def load_stopwords_from_file(path: str) -> Set[str]:
    """
    Loads stopwords from a file that contains one word per line.
    (Use a cleaned .txt version of the provided HTML list.)
    """
    stop = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                stop.add(w)
    return stop


def tokenize(text: str) -> List[str]:
    """
    Tokenize by extracting alphabetic sequences only.
    This automatically removes punctuation and numbers.
    """
    return _WORD_RE.findall(text.lower())


def preprocess_text(text: str, stopwords: Set[str]) -> List[str]:
    """
    Step 1: preprocessing
    - lowercase
    - keep alphabetic tokens only
    - remove stopwords
    """
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in stopwords]
    return tokens


def preprocess_documents(documents: List[Dict], stopwords: Set[str]) -> List[Dict]:
    """
    Applies preprocessing to DOC HEAD and TEXT fields.
    Expects each document dict to have: DOCNO, HEAD, TEXT
    """
    for doc in documents:
        doc["HEAD_TOKENS"] = preprocess_text(doc.get("HEAD", ""), stopwords)
        doc["TEXT_TOKENS"] = preprocess_text(doc.get("TEXT", ""), stopwords)
        # optional: combined field used for indexing
        doc["TOKENS"] = doc["HEAD_TOKENS"] + doc["TEXT_TOKENS"]
    return documents


def preprocess_queries(queries: List[Dict], stopwords: Set[str]) -> List[Dict]:
    """
    Applies preprocessing to query title.
    Expects each query dict to have: num, title
    """
    for q in queries:
        q["TOKENS"] = preprocess_text(q.get("title", ""), stopwords)
    return queries
