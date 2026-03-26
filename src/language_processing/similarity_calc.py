# file for similarity calculations and similar helper functions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime



#JW function for returning keyword for a given query. multiword
# queries will be treated as a vecotr to compare against character name vecotrs.
rp = pd.read_csv("src/language_processing/reverse_postings.csv")
pfc = pd.read_csv("data/piratefolk_comments.csv")

def get_comments_by_character(character):
    row = rp[rp["character"] == character]
    if row.empty:
        return []
    id_string = row.iloc[0]["comment_ids"]
    ids = id_string.split(",")
    
    # get matching comments
    comments = pfc[pfc["id"].isin(ids)]["text"].tolist()
    
    return comments


def build_character_docs():
    character_docs = {}
    for character in rp["character"]:
        comments = get_comments_by_character(character)
        character_docs[character] = " ".join(comments)
    return character_docs

def create_character_tfidf(character_docs):
    characters = list(character_docs.keys())
    docs = list(character_docs.values())

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    return characters, vectorizer, tfidf_matrix

character_docs = build_character_docs()
characters, vectorizer, tfidf_matrix = create_character_tfidf(character_docs)

def query_character(query, vectorizer, tfidf_matrix, characters, top_k=1):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_index = sims.argmax()
    if query in characters:
        return query
    else:
        return characters[best_index]

def make_pickle():
    joblib.dump({
    "matrix": tfidf_matrix,
    "vectorizer": vectorizer,
    "characters": characters
}, "data/model.pkl")
    
#make_pickle()

"""print("below is case sensitive teest")
print(get_comments_by_character("Kuro") == get_comments_by_character("kuro"))
print("below is the query test")
print(query_character("Akainu", vectorizer, tfidf_matrix, characters))
print("enies lobby?")
print([c for c in get_comments_by_character("akainu") if "him" in c.lower()])"""


