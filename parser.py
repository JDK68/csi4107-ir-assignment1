import json
import os
import csv

def parse_document(document_line):
    """
    Parse a single JSON line as a document
    """
    doc = json.loads(document_line)
    parsed_doc = {
        'DOCNO': doc['_id'],
        'HEAD': doc.get('title', 'NO_TITLE'),
        'TEXT': doc.get('text', 'NO_TEXT'),
        'URL': doc.get('metadata', {}).get('url', 'NO_URL')
    }
    return parsed_doc

def parse_documents_from_file(file_path):
    """
    Read the JSON lines file and parse each document
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_docs = [parse_document(line) for line in file]
    return parsed_docs

def parse_query(query_line):
    """
    Parse a single JSON line as a query
    """
    query = json.loads(query_line)
    parsed_query = {
        'num': query['_id'],
        'title': query.get('text', 'NO_TEXT'),
        'query': query.get('metadata', {}).get('query', 'NO_QUERY'),
        'narrative': query.get('metadata', {}).get('narrative', 'NO_NARRATIVE'),
        'url': query.get('metadata', {}).get('url', 'NO_URL')
    }
    return parsed_query

def parse_queries_from_file(file_path):
    """
    Read the JSON lines file and parse each query
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        parsed_queries = [parse_query(line) for line in file]
    return parsed_queries


def parse_documents_from_folder(folder_path):
    """
    Read all files in the specified folder and parse documents from each file
    """
    all_documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            all_documents.extend(parse_documents_from_file(file_path))
    return all_documents

def parse_qrels_from_tsv(file_path):
    """
    Read test.tsv and return relevance judgments (qrels)
    Returns: dict[qid][docid] = score (int)
    """
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row_no, row in enumerate(reader, start=1):
            if not row:
                continue
            # skip possible header
            if row_no == 1 and any("query" in cell.lower() for cell in row):
                continue
            if len(row) < 3:
                raise ValueError(f"Invalid TSV row at line {row_no}: {row}")

            qid = str(row[0]).strip()
            docid = str(row[1]).strip()
            score = int(float(row[2]))

            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = score

    return qrels


def filter_odd_test_queries(parsed_queries):
    """
    Keep only odd query IDs: 1,3,5,...
    Input: list of parsed query dicts (from parse_queries_from_file)
    Output: filtered list
    """
    out = []
    for q in parsed_queries:
        try:
            qid = int(q['num'])
        except Exception:
            out.append(q)  # if non-numeric, keep it
            continue
        if qid % 2 == 1:
            out.append(q)
    return out

def filter_queries_by_qrels_and_odd(parsed_queries, qrels):
    """
    Keep only test queries (present in qrels) AND odd-numbered query IDs.
    parsed_queries: list of dicts with key 'num'
    qrels: dict[qid][docid] = score
    """
    out = []
    for q in parsed_queries:
        qid_str = str(q.get("num", "")).strip()

        # keep only test queries present in qrels
        if qid_str not in qrels:
            continue

        # keep only odd query ids (1,3,5,...)
        try:
            qid = int(qid_str)
        except ValueError:
            out.append(q)
            continue

        if qid % 2 == 1:
            out.append(q)

    # ensure ascending order by query id
    out.sort(key=lambda x: int(x["num"]))
    return out
