"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
from flask import render_template, request
from models import db, Episode, Review
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from language_processing import similarity_calc

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────

data = joblib.load("data/model.pkl")
tfidf_matrix = data["matrix"]
vectorizer = data["vectorizer"]
characters = data["characters"]

character_data = joblib.load("data/character_data.pkl")

def query_character(query):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return characters[sims.argmax()]



def json_search(query):
   
    # only retrieve top 10 relevant documents
    matches = similarity_calc.retrieve_k_docs(query, similarity_calc.tfidf_matrix, 10, similarity_calc.vectorizer, similarity_calc.ids, similarity_calc.docs)
    return json.dumps({
        "name": "Search Results",
        "summary": "",
        "retrieved": matches,            
        "rating": 0,
        "mentions": 0,
        "consensus": "",
        "trend": [],
        "trend_dates": []
        })






def register_routes(app):
    @app.route("/")
    def home():
        return render_template('character-search.html')

    @app.route("/episodes")
    def episodes_search():
        text = request.args.get("title", "")
        return json_search(text)

    @app.route("/characters")
    def character_search():
        return render_template('character-search.html')
    
    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        
        if not query.strip():
            return json.dumps({"error": "empty query"})
        
        result = query_character(query)
        
        print(result)
        return json.dumps({
            "character": result
        })
    @app.route("/csearch")
    def csearch():
        query = request.args.get("q", "").lower().strip()

        if not query:
            return json.dumps({})

        for name in character_data:
            if query in name.lower():
                return json.dumps({name: character_data[name]})

    # fallback (nothing found)
        return json.dumps({})

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
