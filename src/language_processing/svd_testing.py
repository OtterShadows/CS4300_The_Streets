from scipy.sparse.linalg import svds
import os
import joblib
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


matplotlib.use("TkAgg")

#   quick fix for dependency issue
#   may not be the right move, i don't know the code in this file well -DT

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where svd_testing.py lives, language_processing
model_path = os.path.join(current_dir, "data", "model.pkl")
data = joblib.load(model_path)
vectorizer = data["vectorizer"]
td_matrix = data["matrix"]
characters =data["characters"]
#u, s, v_trans = svds(td_matrix, k=100)
print(f"DEBUG: Loaded model.pkl with {len(characters)} characters and td_matrix shape {td_matrix.shape}")
print(f"DEBUG: Sample characters: {characters[:5]}")
print(f"DEBUG: Sample td_matrix row: {td_matrix[0].toarray()[:5]}")


reverse_postings_filename = "reverse_postings_(well).csv"
rp_path = os.path.join(current_dir, "csv", reverse_postings_filename) # get inverted index info
rp = pd.read_csv(rp_path)

piratefolk_comments_filename = "piratefolk_comments_(v2).csv"
pfc_path = os.path.join(current_dir, "csv", piratefolk_comments_filename) #
pfc = pd.read_csv(pfc_path)

"""      print(td_matrix.shape)
print(u.shape)
print(s.shape)
print(v_trans.shape)"""

"""
plt.plot(s[::-1])
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
plt.show()"""

docs_compressed, s, words_compressed = svds(td_matrix, k=45)
words_compressed = words_compressed.transpose()
word_to_index = vectorizer.vocabulary_       
index_to_word = {i:t for t,i in word_to_index.items()}
words_compressed_normed = normalize(words_compressed, axis = 1)
docs_compressed_normed = normalize(docs_compressed)

print(f"DEBUG: docs_compressed_normed shape: {docs_compressed_normed.shape}, words_compressed_normed shape: {words_compressed_normed.shape}")

def make_pickle():
    joblib.dump({
    "svd_words_compressed": words_compressed_normed,
    "s": s,
    "svd_docs_compressed": docs_compressed_normed
}, "src/language_processing/data/svd_model.pkl")
    

def closest_docs_to_query(query_vec_in, k = 5):
    sims = docs_compressed_normed.dot(query_vec_in)
    asort = np.argsort(-sims)[:k+1]
    return [(i, characters[i],sims[i]) for i in asort[1:]]

#editied fubction to return top 5 characters
def closest_doc_to_query(query):
    print(f"DEBUG: Characters from model.pkl: {characters}")
    k=5
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    return [characters[i] for i in asort[1:]]
#print(closest_doc_to_query("potential man"))
            
"""for i, proj, sim in closest_docs_to_query(query_vec):
    doc_svd_vec = docs_compressed_normed[i]
    top_dims_indices = np.argsort(np.abs(doc_svd_vec))[::-1][:3]
    top_dims = [int(dim) for dim in top_dims_indices]
    print("({}, {}, {:.4f}, Top dimensions: {}) ".format(i, proj, sim, top_dims))"""

def display_dims():
    for i in range(45):
        print("Top words in dimension", i)
        dimension_col = words_compressed[:,i].squeeze()
        asort = np.argsort(-dimension_col)
        print([index_to_word[i] for i in asort[:10]])
        print()

def graph_dims(query):
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    query_top_dims_indices = np.argsort(np.abs(query_vec))[::-1][:3]
    query_top_dims = [int(dim) for dim in query_top_dims_indices]
    plt.figure(figsize=(15, 5))
    dimensions = range(len(query_vec))
    colors = ['blue'] * len(query_vec)
    for dim in query_top_dims_indices:
        if dim < len(colors):
            colors[dim] = 'red'

    plt.bar(dimensions, query_vec, color=colors)
    plt.xlabel('Dimension Index')
    plt.ylabel('Dimension Magnitude')
    plt.title(f"Query '{query}' Dimensions (Top 3 highlighted in red)")
    plt.xticks(dimensions[::5])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

#display_dims()
"""
graph_dims("Biggest bum")
graph_dims("Plants")
graph_dims("Luffy Katakuri fight")
graph_dims("biggest coward")
graph_dims("Gear 5")"""

#make_pickle()
# make_pickle()

    




# context: appropriate the SVD model made for character docs for retrieval/ranking of comments,
# hopefully ones that match the meaning of the query better...
# for use in routes.py
# output: list of tuples of form (comment_text, sim_score)
def svd_retrieve_k_sim_comments(character: str, query: str, tfidf_matrix, k = 20):
    # get all comments mentioning character from reverse postings
    print(f"\033[95mUsing SVD retrieval for character '{character}' and query '{query}'\033[0m")
    row = rp[rp["character"] == character]
    if row.empty:
        return []

    ids_string = row.iloc[0]["comment_ids"] # comma separated string of ids
    ids_list = ids_string.split(",")

    matched_df = pfc[pfc["id"].isin(ids_list)].copy()
        # filter to only have ids in ids_list
    if matched_df.empty:
        return []

    indices = row.index.tolist()
    comment_tfidf_matrix = vectorizer.transform(matched_df["text"].fillna("")).toarray()
    comments_compressed = normalize(comment_tfidf_matrix.dot(words_compressed))

    query_vec = vectorizer.transform([query])
    query_vec_comp = normalize(query_vec.toarray().dot(words_compressed)).squeeze()
    sims = comments_compressed.dot(query_vec_comp)
    ranked = np.argsort(-sims)[:k]
    print(f"DEBUG: In retrieval function, len(ranked) = {len(ranked)} for character '{character}' and query '{query}'")

    result_tuples = [(matched_df.iloc[i]["id"], float(sims[i])) for i in ranked]
    return result_tuples





# UNUSED? -------------------------------------------------------------------------------
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]



# TEST -----------------------------------------------------------------------------------

# query =" potential man"
# query_tfidf = vectorizer.transform([query]).toarray()
# query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()