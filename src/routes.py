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

    matches = similarity_calc.retrieve_k_docs(query, similarity_calc.tfidf_matrix, 10, similarity_calc.vectorizer, similarity_calc.ids, similarity_calc.docs)
    summary = "Summary..."
    retrieved = matches

    # character_score = similarity_calc.get_character_rating(name)

    print("\033[32m" + "Calculating trend_data..." + "\033[0m")
    trend_data = similarity_calc.get_star_rating_over_time(name, 5)
    print("\033[32m" + "Calculating trend_stars..." + "\033[0m")
    trend_stars = [round(v, 2) for v in trend_data.values()]
    print("\033[32m" + "Calculating trend_dates..." + "\033[0m")
    trend_dates = list(trend_data.keys())


    print("\033[32m" + "Calculating rating..." + "\033[0m")
    rating = sum(trend_stars) / len(trend_stars)
    print("\033[32m" + "Rating: " + str(rating) + "\033[0m")

    mentions = similarity_calc.num_mentions(name)
    # mentions = 0
    if rating >= 6:
        consensus = "Positive"
    elif rating <= 4:
        consensus = "Negative"
    else:
        consensus = "Neutral"

    return json.dumps({
        "name": name,
        "summary": summary,
        "retrieved": matches,
        "rating": rating,
        "mentions": mentions,
        "consensus": consensus,
        "trend": trend_stars,
        "trend_dates": trend_dates
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
