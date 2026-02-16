# CSI 4107: Information Retrieval and the Internet
## Assignment 1 – README

## Team Information
- Name: Jad Kebal, Student number: 300329290
- Name: Gini Sheja, Student number: 300222140
- Name: Brooklyn McClelland, Student number: 300311745
- Name: Solin Maaroof, Student number: 300250903

## Task Division
- **Jad Kebal - Dataset + Preprocessing**:
  - Parse corpus/queries from .jsonl files
  - Stopword list, tokenization, stopword removal

- **Gini Sheja - Indexing + README**:
  - Build the inverted index structure
  - Store TF per document per term
  - Write README

- **Brooklyn McClelland - Retrieval & Ranking + Integration**:
  - Cosine similarity scoring
  - Main pipeline script tying all steps together

- **Solin Maaroof - Indexing + README**:
  - Compute and store document frequencies (DF)
  - Compute and store document lengths
  - Run experiments + Results file + trec_eval + README

## Dataset

We use the **Scifact** dataset from the BEIR collection, as required.

Dataset files (not included in the repository):
- `corpus.jsonl`
- `queries.jsonl`
- `qrels/test.tsv`

Dataset statistics:
- Number of documents: **5183**
- Number of raw queries: **1109**
- Number of test queries (from qrels): **300** (odd IDs only: 1, 3, 5...)

## Functionality of Programs 
### Step 1 - Preprocessing
The preprocessing module transforms raw document and query text into clean tokens suitable for indexing.

The following steps are implemented:

1. Lowercasing
2. Tokenization using a regular expression that keeps **alphabetic tokens only**
   - This removes punctuation and numeric tokens
3. Stopword removal using a predefined stopword list (`stopwords.txt`)
4. The same preprocessing pipeline is applied to:
   - Document titles
   - Document body text
   - Query titles

#### Output
For each document:
- `HEAD_TOKENS`: tokens extracted from the document title
- `TEXT_TOKENS`: tokens extracted from the document body
- `TOKENS`: concatenation of title and body tokens

For each query:
- `TOKENS`: preprocessed query tokens

#### Example (Document)
Document ID: `4983`

- HEAD_TOKENS (first 20): ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging']

- TEXT_TOKENS (first 20): ['alterations', 'architecture', 'cerebral', 'white', 'matter', 'developing', 'human', 'brain', 'affect', 'cortical', 'development', 'result', 'functional', 'disabilities', 'line', 'scan', 'diffusion', 'weighted', 'magnetic', 'resonance']

### Step 2 – Indexing
#### Algorithm and Data Structures
An **inverted index** is built from the preprocessed tokens.

The following data structures are used:

- **Inverted index**
inverted_index[term][doc_id] = tf
where `tf` is the term frequency of the term in the document.

- **Document frequency**
doc_freq[term] = number of documents containing the term

- **Document length**
doc_lengths[doc_id] = number of tokens in the document

Python dictionaries are used to ensure fast access to postings lists.

#### Vocabulary Size
After preprocessing and indexing:
- Vocabulary size |V| = **29,953** unique terms

#### Sample of 100 Tokens
Example sample from the vocabulary: ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging', 'alterations', 'architecture', 'developing', 'brain', 'affect', 'cortical', 'result', 'functional', 'disabilities', 'line', 'scan', 'weighted', 'mri', 'sequence', 'analysis', 'applied', 'measure', 'apparent', 'coefficient', 'calculate', 'relative', 'anisotropy', 'delineate', 'dimensional', 'fiber', 'preterm', 'full', 'term', 'infants', 'assess', 'effects', 'prematurity', 'early', 'gestation', 'studied', 'central', 'mean', 'wk', 'microm', 'decreased', 'posterior', 'limb', 'internal', 'capsule', 'coefficients', 'versus', 'closer', 'birth', 'absolute', 'values', 'areas', 'compared', 'nonmyelinated', 'fibers', 'corpus', 'callosum', 'visible', 'marked', 'differences', 'organization', 'data', 'indicate', 'quantitative', 'assessment', 'water', 'insight', 'living', 'induction', 'myelodysplasia', 'myeloid', 'derived', 'suppressor', 'cells', 'myelodysplastic', 'syndromes', 'mds', 'age', 'dependent', 'stem', 'cell', 'malignancies', 'share', 'biological', 'features', 'activated', 'adaptive']

#### Example Index Entry
Term: `microstructural`

- Document frequency (DF): **6**
- Example postings (doc_id → tf):
First 5 postings (doc_id -> tf): [('4983', 2), ('1412089', 1), ('1472815', 1), ('3205945', 8), ('22107641', 3)]

### Step 3 – Retrieval and Ranking
The retrieval and ranking module uses the inverted index to calculate cosine similarity scores between a query and each document  

The following steps are implemented:

1. Retrieving documents containing at least one query word - other documents are discarded
2. Calculating the tf, idf, and norms, and using these values to calculate the cosine similarity score
3. Returning the ranked doc ids with their cosine similarity scores

#### Output
For each query:
- `ranked_docs`: a list of document ids with their cosine similarity scores

### Instructions
1. **Prerequisite:** Place the Scifact dataset in folder `scifact/` at the project root (or under `datasets/scifact/`). Required files: `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`.
2. Ensure `stopwords.txt` is present (one word per line).
3. **Verify Steps 1 and 2:**
```bash
python check_step1_step2.py
```
4. **Run experiments and generate the Results file:**
```bash
python run_experiments.py
```
   This script runs two configurations (title-only and title+text index), writes TREC-format files `Results_title_only.txt`, `Results_title_and_text.txt`, and the best run to `Results` (for submission). It computes Mean Average Precision (MAP).

### Experiments and Results

#### Runs: Title-only vs Title+Text
As required, we ran the system in two configurations:
- **Run 1 (title only):** Index built from document titles only (`HEAD_TOKENS`). Queries are unchanged (title text).
- **Run 2 (title and text):** Index built from titles and full document text (`TOKENS` = `HEAD_TOKENS` + `TEXT_TOKENS`).

#### Mean Average Precision (MAP)
MAP was computed using our Python implementation (equivalent to trec_eval’s MAP).

| Run            | MAP        |
|----------------|------------|
| Title only     | 0.3697     |
| Title + text   | **0.5018** |

**Results:**
 - The **title+text** run gives better results (MAP 0.5018 vs 0.3697 for title only). 
 - Indexing only titles yields a smaller vocabulary and fewer matches: many relevant documents contain the query terms in the abstract/body but not in the title, so they are missed or ranked lower. 
 - With titles and full text, the system can match queries to documents that discuss the claim in the body even when the title is generic. 
 - For Scifact, the full text is essential for relevance, so we use the title+text run as our best run and submit its results file.

#### First 10 results for the first 2 queries
**Query 1:**  
1. 13231899 (0.0803)  
2. 10931595 (0.0747)  
3. 10608397 (0.0737)  
4. 10607877 (0.0715)  
5. 31543713 (0.0643)  
6. 25404036 (0.0533)  
7. 9580772 (0.0521)  
8. 24998637 (0.0510)  
9. 16939583 (0.0502)  
10. 6863070 (0.0493)  

**Query 3:**  
1. 23389795 (0.3699)  
2. 2739854 (0.3317)  
3. 14717500 (0.2598)  
4. 4632921 (0.2275)  
5. 8411251 (0.2021)  
6. 4378885 (0.1582)  
7. 32181055 (0.1526)  
8. 3672261 (0.1505)  
9. 14019636 (0.1463)  
10. 4414547 (0.1419)  

For query 3, the relevant document 14717500 (from qrels) appears at rank 3, which illustrates that the ranking is putting relevant documents near the top.