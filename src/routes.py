"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
from flask import render_template, request
from models import db, Episode, Review
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────

data = joblib.load("data/model.pkl")
tfidf_matrix = data["matrix"]
vectorizer = data["vectorizer"]
characters = data["characters"]

def query_character(query):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return characters[sims.argmax()]



def json_search(query):
    """if not query or not query.strip():
        query = "Kardashian"
    results = db.session.query(Episode, Review).join(
        Review, Episode.id == Review.id
    ).filter(
        Episode.title.ilike(f'%{query}%')
    ).all()
    matches = []
    for episode, review in results:
        matches.append({
            'title': episode.title,
            'descr': episode.descr,
            'imdb_rating': review.imdb_rating
        })
    return json.dumps(matches)"""




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

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
