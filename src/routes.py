"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import os # (for loading the data/model.pkl) (TODO add to requirements.txt or no?)
import json
from flask import render_template, request
from models import db, Episode, Review
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from language_processing import similarity_calc
from language_processing import character_class

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where routes.py lives
model_path = os.path.join(current_dir, "language_processing", "data", "model.pkl")

# data = joblib.load("data/model.pkl")
data = joblib.load(model_path)
tfidf_matrix = data["matrix"]
vectorizer = data["vectorizer"]
characters = data["characters"]

# character_data = joblib.load("data/character_data.pkl")
character_data_path = os.path.join(current_dir, "language_processing", "data", "character_data.pkl")
character_data = joblib.load(character_data_path)

# calculates similarity between query and character docs, returns best match's name
def query_character(query):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return characters[sims.argmax()]
# ... not sure what this function was for? anyways, it's outdated since it calls a
# function that doesn't exist
# def json_search(query):
   
#     # only retrieve top 10 relevant documents
#     matches = similarity_calc.retrieve_k_docs(query, similarity_calc.tfidf_matrix, 10, similarity_calc.vectorizer, similarity_calc.ids, similarity_calc.docs)
#     return json.dumps({
#         "name": "Search Results",
#         "summary": "",
#         "retrieved": matches,            
#         "rating": 0,
#         "mentions": 0,
#         "consensus": "",
#         "trend": [],
#         "trend_dates": []
#         })



def register_routes(app):
    @app.route("/")
    def home():
        return render_template('character-search.html')
    
    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        
        if not query.strip():
            return json.dumps({"error": "empty query"})
        
        # calculate the similarity of the query with the character "docs" and 
        # return the most similar character
        result = similarity_calc.query_character(query, 
            similarity_calc.vectorizer,
            similarity_calc.tfidf_matrix,
            similarity_calc.characters)
        print(f"Received search query: '{query}' -> matched character: '{result}'")

        # calculate top k relevant comments
        relevant_comments = similarity_calc.retrieve_k_sim_comments(
            query = query,
            vectorizer = similarity_calc.comment_term_vectorizer,
            comment_term_tfidf_matrix = similarity_calc.comment_term_tfidf_matrix,
            ids = similarity_calc.comment_ids,
            texts = similarity_calc.texts,
            k = 25
        ) # should return list of tuples of form (id, sim_score)

        comment_list = [] # list of relevant Comment objects, where "Comment" defined in character_class.py
        for (id, score) in relevant_comments:
            comment_list.append(character_class.create_comment(id, score))

        return json.dumps({
            "character": result, # string of most similar character to query
            "relevant_comments": [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality, "sim_score": c.sim_score} for c in comment_list]
        })
    
    @app.route("/csearch")
    def csearch():
        name = request.args.get("q", "")
        print(f"Received Csearch query: '{name}'")
        if not name:
            return json.dumps({})
        if name in character_data.keys():
            print(f"Exact match found for {name}")
            return json.dumps(character_data[name])
        print(f"{name} is not a character name")

    # fallback (nothing found)
        return json.dumps({})

    # if USE_LLM:
    #     from llm_routes import register_chat_route
    #     # register_chat_route(app, json_search)
