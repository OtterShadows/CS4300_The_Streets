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
import datetime



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

print("below is case sensitive teest")
print(get_comments_by_character("Kuro") == get_comments_by_character("kuro"))
print("below is the query test")
print(query_character("Akainu", vectorizer, tfidf_matrix, characters))
print("enies lobby?")
print([c for c in get_comments_by_character("akainu") if "him" in c.lower()])


# Helper to match_name
def edit_distance(source: str, target: str):
    D = np.zeros((len(source) + 1, len(target) + 1))
    for i in range(len(source) + 1):
        D[i, 0] = i
    for j in range(len(target) + 1):
        D[0, j] = j
    D[0,0] = 0

    for i in range(1, len(source) + 1):
        for j in range(1, len(target) + 1):
            deletion = D[i-1, j] + 1
            insertion = D[i, j-1] + 1

            if target[j - 1] == source[i - 1]:
                substitution = D[i-1, j-1] + 0
            else: 
                substitution = D[i-1,j-1] + 2
            D[i,j] = min(deletion, insertion, substitution)
    return D[len(source), len(target)]







# Helper: turn txt file into list[str] with each element representing a line
def file_to_list_stripped(file_path: str) -> list[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]



# Function 1: Match name using edit distance
# 	- Need to know user if trying to search a name, and i don't think
# 	there's a clean way to determine unless we make a checkbox saying:
# 	"Search for character"
# 	- Input example: "Loofy"
# 	- Output example: "Luffy"
def match_name(input_name: str, character_list: list[str]):
    best_edit_distance = 1000000
    best_match = ""
    for character in character_list:
        char_edit_dist = edit_distance(input_name.lower(), character.lower())
        if char_edit_dist < best_edit_distance:
            best_match = character
            best_edit_distance = char_edit_dist
    return best_match;

# turn csv file with (character, count) and converts into list of character names
def load_character_names(csv_path: str):
    df = pd.read_csv(csv_path)
    return df["character"].tolist()

char_list = load_character_names("data/character_names.csv")


# Current problem: Some characters have aliases. Should we use the aliases for later searching documents?
# Should we add aliases to the current list? Would be hard to do in a systematic way...



# Function 1.5: Create tf-idf matrix from docs
def create_tfidf_matrix(filepath: str):
    df = pd.read_csv(filepath)

    df["text"] = df["text"].fillna("").astype(str)

    docs = df["text"].tolist()
    ids = df["id"].tolist()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)
    return (ids, vectorizer, tfidf_matrix, docs)
    
ids, vectorizer, tfidf_matrix, docs = create_tfidf_matrix("data/piratefolk_comments.csv")

# Function 2: Return k most relevant documents for query
# 	- Assume "Search for character" checkbox is not checked. Then:
# 	- Input example: "Most useless in Whole Cake arc"
# 	- Output: Ranked comments based off cosine similarity
def retrieve_k_docs(query, td_matrix, k, vectorizer, ids, docs):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, td_matrix).flatten()

    # get just top k indeces
    top_indices = similarities.argsort()[::-1][:k]

    rankings = []
    for i in top_indices:
        # docs can be either list or DataFrame
        if isinstance(docs, pd.DataFrame):
            text = docs.iloc[i]["text"]  # pandas row
        else:
            text = docs[i]  # already a list

        rankings.append({
            "id": ids[i],
            "score": float(similarities[i]),
            "text": text
        })



# Helper function: match if there is less than k edit distance from some term
def fuzzy_term_match(query, document, k):
    query = query.lower()
    terms = document.lower().split()

    for term in terms:
        if edit_distance(query, term) < k:
            return 1
    return 0

    # problem: wouldn't work for names longer than a space



#nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

start_of_dataset_timestamp = 1678648020
end_of_dataset_timestamp = 1741543624

docs = pd.read_csv("data/piratefolk_comments.csv")
comments = docs.dropna(subset=["text"]).to_dict("records")



# Function 3: Return character rating
# 	- Input: Character name
# 	- Output: Score of character, -1 to 1
# 	- Use similarity with good / bad document
# first implementation - relies on sentiment analyzer
def get_character_rating(character: str, start_timestap=start_of_dataset_timestamp, end_timestamp=end_of_dataset_timestamp):
    # is there a more elegant / involved way than looping through comments?
    scores = []
    for comment in comments:
        comment_text = comment["text"]
        if fuzzy_term_match(character, comment_text, 3):
            score = sid.polarity_scores(comment_text)["compound"] # get compound score for doc
            score_weighted = score * comment["score"]
            scores.append(score_weighted)
    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores)

def to_star_rating(raw_score: float) -> float:
    if raw_score >= 0:
        curved_score = (raw_score ** 0.3)
    else:
        curved_score = -1 * ((-1 * raw_score) ** 0.3)
    num_stars = (curved_score + 1) * 5
    if num_stars < 0:
        num_stars = 0
    elif num_stars > 10:
        num_stars = 10
    return num_stars

# for popularity trend graph, splits interval into k parts and get charater rating for each part
def get_star_rating_over_time(character: str, k: int, start_timestamp=start_of_dataset_timestamp, end_timestamp=end_of_dataset_timestamp):
    interval = (end_timestamp - start_timestamp) // k
    scores = {}
    for i in range(k):
        sub_interval_start = start_timestamp + i * interval
        sub_interval_end = end_timestamp if i == k - 1 else sub_interval_start + interval
        score = get_character_rating(character, start_timestap=sub_interval_start, end_timestamp=sub_interval_end)
        print(f"\033[95mScore for {character} from {datetime.fromtimestamp(sub_interval_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(sub_interval_end).strftime('%Y-%m-%d')}: {score}\033[0m")
        stars = to_star_rating(score)
        print(f"\033[95mStar rating for {character} from {datetime.fromtimestamp(sub_interval_start).strftime('%Y-%m-%d')} to {datetime.fromtimestamp(sub_interval_end).strftime('%Y-%m-%d')}: {stars}\033[0m")
        date_object = datetime.fromtimestamp(sub_interval_start)
        date_formatted = date_object.strftime("%Y-%m-%d")
        scores[date_formatted] = stars
    print(f"\033[34mStar ratings for {character} over time: {scores}\033[0m")
    return scores

def get_star_rating_average(scores: dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)
    
    return (raw_score + 1) * 5



# TEST FUNCTIONS --------------------------------------------------------
def get_character_rating_test():
    test_names = ["luffy", "luffe", "nami", "kuma", "shanks"]
    for name in test_names:
        print(f"{name}: {to_star_rating(get_character_rating(name))}")

def retrieve_k_docs_test():
    csv_path = "data/piratefolk_comments.csv"
    query = "bum wano"
    ids, vectorizer, tfidf_matrix, docs = create_tfidf_matrix(csv_path)
    rankings = retrieve_k_docs(query, tfidf_matrix, 10, vectorizer, ids, docs)
    for ranking in rankings:
        print(ranking["text"])



# second implementation - should compare document similarities to "good" and "bad" word documents





#retrieve_k_docs_test()