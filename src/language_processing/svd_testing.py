from scipy.sparse.linalg import svds
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# matplotlib.use("TkAgg")

#   quick fix for dependency issue
#   may not be the right move, i don't know the code in this file well -DT

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where svd_testing.py lives, language_processing
model_path = os.path.join(current_dir, "data", "model.pkl")
data = joblib.load(model_path)
vectorizer = data["vectorizer"]
td_matrix = data["matrix"]
characters =data["characters"]
#u, s, v_trans = svds(td_matrix, k=100)

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

def closest_doc_to_query(query):
    k=5
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    return characters[asort[1]]
# print(closest_doc_to_query("potential man"))
            
"""for i, proj, sim in closest_docs_to_query(query_vec):
    doc_svd_vec = docs_compressed_normed[i]
    top_dims_indices = np.argsort(np.abs(doc_svd_vec))[::-1][:3]
    top_dims = [int(dim) for dim in top_dims_indices]
    print("({}, {}, {:.4f}, Top dimensions: {}) ".format(i, proj, sim, top_dims))"""

make_pickle()

    




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