"""
Routes: home page and episode search.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for LLM specific routes.
"""
import json
from flask import render_template, request
from models import db, Episode, Review
from language_processing import similarity_calc

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────


def json_search(query):
    if not query or not query.strip():
        query = "Luffy"
    print("\033[32m" + "Query: " + query + "\033[0m")
    name = similarity_calc.match_name(query, similarity_calc.char_list)
    print("\033[32m" + "Name: " + name + "\033[0m")
    matches = similarity_calc.retrieve_k_docs(name, similarity_calc.tfidf_matrix, 10, similarity_calc.vectorizer, similarity_calc.ids, similarity_calc.docs)
    summary = "Summary..."
    retrieved = matches

    character_score = similarity_calc.get_character_rating(name)
    rating = similarity_calc.to_star_rating(character_score)
    print("\033[32m" + "Rating: " + str(rating) + "\033[0m")

    rating = f"{float(rating):.1f}"
    return json.dumps({
        "name": query,
        "summary": summary,
        "retrieved": retrieved,
        "rating": rating
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

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
