import math

def compute_similarity_scores(query, inverted_index, docs):
    """
    Step 3: Use the inverted index to find documents containing at least one query word. 
    Compute the cosine similarity scores between a query and each document. 

    Returns:
      - sim_values: query -> doc_id (Similarity values between the query and each document)
      - *********Rank the documents in decreasing order of similarity scores.
    """

    # find documents containing at least one query word
    tokens = query["TOKENS"]
    docs_with_query = set()
    for token in tokens:
        if token in inverted_index:
            docs_with_query.update(inverted_index[token].keys())
              

    # calculate doc_max_tf, used for normalization
    doc_max_tf = {}
    for term, postings in inverted_index.items():
        for doc_id, freq in postings.items():
            if doc_id not in doc_max_tf or freq > doc_max_tf[doc_id]:
                doc_max_tf[doc_id] = freq

    N = len(docs)
    doc_sum_squares = {} 
    query_weights = {}

    # calculate doc norms
    for term, postings in inverted_index.items():
        # calculate the idf for this term
        df = len(postings)
        idf = math.log2(N/df)

        if term in tokens:
            query_weights[term] = idf

        for doc_id, tf in postings.items():
            # calculate the weight of the term in this doc
            max_f = doc_max_tf.get(doc_id, 1)
            normalized_tf = tf/max_f
            weight = normalized_tf * idf

            # add the square of the weight to the doc's running total
            doc_sum_squares[doc_id] = doc_sum_squares.get(doc_id, 0) + weight**2
    
    # take the square root of the doc_sum_squares to get the doc_norm
    doc_norms = {}
    for doc_id, total_sum in doc_sum_squares.items():
        doc_norms[doc_id] = math.sqrt(total_sum)

    query_norm = 0
    for token in tokens:
        
        query_norm = query_norm + float(query_weights.get(token, 0))**2
        
    query_norm = math.sqrt(query_norm)

    sim_values = {}

    for doc_id in docs_with_query:
        dot_product = 0
        for token in tokens:
            query_weight = query_weights.get(token)
            d_tf = inverted_index[token].get(doc_id, 0)
            doc_weight = (d_tf / doc_max_tf[doc_id]) * query_weight
            dot_product = dot_product + (doc_weight * query_weight)
            sim_value = dot_product / (query_norm * doc_norms[doc_id])
            sim_values[doc_id] = sim_value

    # return the ranked doc ids with their cosine similarity scores
    ranked_docs = sorted(sim_values.items(), key=lambda x: x[1], reverse=True)
    
    return ranked_docs