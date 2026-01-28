# CSI4107 – Information Retrieval  
## Assignment 1 – Progress Report (Step 1 & Step 2)

---

## Team Information

- Name: Jad Kebal
- Student number: 300329290
- Name:  
- Student number: 
- Name:  
- Student number: 
- Name:  
- Student number: 

### Task Division
- **Jad Kebal**:
  - Step 1: Preprocessing
  - Step 2: Indexing
  - Dataset setup (Scifact)
  - Sanity checks and verification scripts


---

## Dataset

We use the **Scifact** dataset from the BEIR collection, as required.

Dataset files (not included in the repository):
- `corpus.jsonl`
- `queries.jsonl`
- `qrels/test.tsv`

Dataset statistics:
- Number of documents: **5183**
- Number of raw queries: **1109**
- Number of test queries (from qrels): **300** (odd IDs only: 1, 3, 5, …)

---

## Step 1 – Preprocessing (Implemented)

### Functionality
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

### Output
For each document:
- `HEAD_TOKENS`: tokens extracted from the document title
- `TEXT_TOKENS`: tokens extracted from the document body
- `TOKENS`: concatenation of title and body tokens

For each query:
- `TOKENS`: preprocessed query tokens

### Example (Document)
Document ID: `4983`

- HEAD_TOKENS (first 20): ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging']

- TEXT_TOKENS (first 20): ['alterations', 'architecture', 'cerebral', 'white', 'matter', 'developing', 'human', 'brain', 'affect', 'cortical', 'development', 'result', 'functional', 'disabilities', 'line', 'scan', 'diffusion', 'weighted', 'magnetic', 'resonance']

---

## Step 2 – Indexing (Implemented)

### Algorithm and Data Structures
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

### Vocabulary Size
After preprocessing and indexing:
- Vocabulary size |V| = **29,953** unique terms

### Sample of 100 Tokens
Example sample from the vocabulary: ['microstructural', 'development', 'human', 'newborn', 'cerebral', 'white', 'matter', 'assessed', 'vivo', 'diffusion', 'tensor', 'magnetic', 'resonance', 'imaging', 'alterations', 'architecture', 'developing', 'brain', 'affect', 'cortical', 'result', 'functional', 'disabilities', 'line', 'scan', 'weighted', 'mri', 'sequence', 'analysis', 'applied', 'measure', 'apparent', 'coefficient', 'calculate', 'relative', 'anisotropy', 'delineate', 'dimensional', 'fiber', 'preterm', 'full', 'term', 'infants', 'assess', 'effects', 'prematurity', 'early', 'gestation', 'studied', 'central', 'mean', 'wk', 'microm', 'decreased', 'posterior', 'limb', 'internal', 'capsule', 'coefficients', 'versus', 'closer', 'birth', 'absolute', 'values', 'areas', 'compared', 'nonmyelinated', 'fibers', 'corpus', 'callosum', 'visible', 'marked', 'differences', 'organization', 'data', 'indicate', 'quantitative', 'assessment', 'water', 'insight', 'living', 'induction', 'myelodysplasia', 'myeloid', 'derived', 'suppressor', 'cells', 'myelodysplastic', 'syndromes', 'mds', 'age', 'dependent', 'stem', 'cell', 'malignancies', 'share', 'biological', 'features', 'activated', 'adaptive']

### Example Index Entry
Term: `microstructural`

- Document frequency (DF): **6**
- Example postings (doc_id → tf):
First 5 postings (doc_id -> tf): [('4983', 2), ('1412089', 1), ('1472815', 1), ('3205945', 8), ('22107641', 3)]

---

## Verification and How to Run

A verification script is provided to validate Step 1 and Step 2:


### Instructions
1. Place the Scifact dataset under: datasets/scifact/
2. Ensure `stopwords.txt` is present (one word per line).
3. Run:
```bash
python check_step1_step2.py
