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
from language_processing import svd_testing
# from pathlib import Path
import requests
from functools import lru_cache
from language_processing import character_class
from sklearn.preprocessing import normalize
from language_processing import character_counts

# ── AI toggle ──
USE_LLM = False
# USE_LLM = True
# ───────────────

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where routes.py lives
model_path = os.path.join(current_dir, "language_processing", "data", "model.pkl")
svd_model_path = os.path.join(current_dir, "language_processing", "data", "svd_model.pkl")

# data = joblib.load("data/model.pkl")
data = joblib.load(model_path)
tfidf_matrix = data["matrix"]
vectorizer = data["vectorizer"]
characters = data["characters"]
svd_data = joblib.load(svd_model_path)
svd_words_compressed = svd_data["svd_words_compressed"]
svd_docs_compressed = svd_data["svd_docs_compressed"]

comments_df, postings_df = character_class.load_data()
# ===== One Piece GraphQL API Setup =====
ONE_PIECE_API_URL = "https://onepieceql.com/api/graphql"
_api_cache = {}  # Cache for API data

def fetch_all_characters():
    """Fetch all One Piece characters from the GraphQL API with pagination"""
    global _api_cache
    
    if _api_cache:  # Use cached data if available
        return _api_cache
    
    try:
        page = 1
        limit = 100  # Fetch 100 at a time
        total_fetched = 0
        
        # GraphQL query with pagination
        query_template = """
        {{
          characters(filter: {{ limit: {limit}, page: {page} }}) {{
            results {{
              name
              image
            }}
            info {{
              count
              pages
            }}
          }}
        }}
        """
        
        while True:
            query = query_template.format(limit=limit, page=page)
            response = requests.post(
                ONE_PIECE_API_URL,
                json={"query": query},
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"⚠ API error on page {page}: {response.status_code}")
                break
            
            data = response.json()
            if 'errors' in data:
                print(f"⚠ GraphQL error on page {page}: {data['errors']}")
                break
            
            if 'data' not in data or not data['data']['characters']:
                break
            
            chars_response = data['data']['characters']
            characters = chars_response.get('results', [])
            
            # Add characters to cache
            for char in characters:
                name = char.get('name', '').strip()
                if name:
                    _api_cache[name] = char
            
            total_fetched += len(characters)
            info = chars_response.get('info', {})
            total_count = info.get('count', 0)
            
            print(f"✓ Loaded {len(characters)} characters from page {page} (Total: {total_fetched}/{total_count})")
            
            # Check if we got all characters
            if total_fetched >= total_count or not characters:
                break
            
            page += 1
        
        print(f"✓ Successfully loaded {len(_api_cache)} total characters from One Piece GraphQL API")
        return _api_cache
        
    except Exception as e:
        print(f"⚠ Failed to fetch from One Piece GraphQL API: {e}")
    
    return {}

@lru_cache(maxsize=128)
def get_character_image(character_name):
    """
    Fetch character image from the official One Piece GraphQL API.
    Supports multiple name formats and falls back to placeholder if not found.
    """
    # Fetch API data first (cached after first load)
    characters = fetch_all_characters()
    
    # Try exact match first
    if character_name in characters:
        char_data = characters[character_name]
        if 'image' in char_data and char_data['image']:
            url = char_data['image']
            print(f"✓ Found image for {character_name}")
            return url
    
    # Try fuzzy matching for name variations
    # Remove spaces and dots for comparison
    query_normalized = character_name.replace(" ", "").replace(".", "").lower()
    for api_name, char_data in characters.items():
        if query_normalized == api_name.replace(" ", "").replace(".", "").lower():
            if 'image' in char_data and char_data['image']:
                print(f"✓ Found image for {character_name} (matched: {api_name})")
                return char_data['image']
    
    # Fallback to placeholder
    placeholder = f"https://via.placeholder.com/400x500/0066cc/ffffff?text={character_name.replace(' ', '%20')}"
    print(f"⚠ No image found for {character_name}, using placeholder")
    return placeholder

# Auto-generate character_data.pkl if it doesn't exist
character_data_path = os.path.join(current_dir, "language_processing", "data", "character_data.pkl")

def ensure_character_data_exists():
    """Generate character_data.pkl if it doesn't exist."""
    if not os.path.exists(character_data_path):
        print("🔄 Generating character_data.pkl...")
        try:
            characters = character_class.create_all_characters(postings_df, comments_df)
            char_dict = character_class.characters_to_dict(characters)
            joblib.dump(char_dict, character_data_path)
            print(f"✅ Successfully generated character_data.pkl with {len(char_dict)} characters")
            return char_dict
        except Exception as e:
            print(f"❌ Error generating character_data.pkl: {e}")
            raise
    return None

# Ensure character data exists before loading
ensure_character_data_exists()
character_data = joblib.load(character_data_path)

# calculates similarity between query and character docs, returns best match's name
def query_character(query):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return characters[sims.argmax()]


def register_routes(app):
    @app.route("/")
    def home():
        return render_template('character-search.html')
    
    @app.route("/search")
    def search():
        query = request.args.get("q", "")
        
        if not query.strip():
            return json.dumps({"error": "empty query"})
        
        #first check if the query matches a character name (with fuzzy matching)
        if character_counts.fuzzy_match_character(query, character_counts.names_and_variants) != "":
            result = character_counts.fuzzy_match_character(query, character_counts.names_and_variants)
        # calculate the similarity of the query with the character "docs" and 
        # return the most similar character
        else:
            result = svd_testing.closest_doc_to_query(query)
        print(f"Received search query: '{query}' -> matched character: '{result}'")

        # calculate top k relevant comments
        # relevant_comments = similarity_calc.retrieve_k_sim_comments(
        #     query = query,
        #     vectorizer = similarity_calc.comment_term_vectorizer,
        #     comment_term_tfidf_matrix = similarity_calc.comment_term_tfidf_matrix,
        #     ids = similarity_calc.comment_ids,
        #     texts = similarity_calc.texts,
        #     k = 1000
        # ) # should return list of tuples of form (id, sim_score)

        relevant_comments = similarity_calc.newer_retrieve_k_sim_comments(
            character = result,
            query = query,
            comment_term_vectorizer = similarity_calc.comment_term_vectorizer,
            k = 50
        )


        relevant_comments_containing_character = similarity_calc.prioritize_comments_by_character(result, relevant_comments)

        comment_list = [] # list of relevant Comment objects, where "Comment" defined in character_class.py
        for (id, score) in relevant_comments:
            c = character_class.create_comment(id, score, comments_df)
            if c is not None:
                comment_list.append(c)

        comments_json = [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality, "sim_score": c.sim_score} for c in comment_list]
        print(f"DEBUG: Returning {len(comments_json)} comments for query '{query}' matched to '{result}'")
        if len(comments_json) > 0:
            print(f"DEBUG: First comment: user={comments_json[0].get('user')}, text_preview={comments_json[0].get('text')[:50] if comments_json[0].get('text') else 'N/A'}")
        
        return json.dumps({
            "character": result, # string of most similar character to query
            "relevant_comments": comments_json
        })
    
    @app.route("/csearch")
    def csearch():
        name = request.args.get("q", "")
        print(f"Received Csearch query: '{name}'")
        if not name:
            return json.dumps({})
        if name in character_data.keys():
            print(f"Exact match found for {name}")
            # Get character data and add image URL
            char_info = character_data[name]
            
            # Convert to dict if it's not already
            if isinstance(char_info, dict):
                response_data = char_info.copy()
            else:
                # If it's an object, convert to dict
                response_data = vars(char_info) if hasattr(char_info, '__dict__') else {'data': str(char_info)}
            
            # Add image URL
            response_data['image_url'] = get_character_image(name)
            print(f"Returning character data with image_url: {response_data.get('image_url')}")
            return json.dumps(response_data, default=str)
        print(f"{name} is not a character name")
        # fallback (nothing found)
        return json.dumps({})

    # if USE_LLM:
    #     from llm_routes import register_chat_route
    #     # register_chat_route(app, json_search)
