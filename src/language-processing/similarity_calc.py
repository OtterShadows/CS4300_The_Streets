# file for similarity calculations and similar helper functions
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer



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

def create_character_docs():
    character_docs = {}
    for character in rp["character"]:
        comments = get_comments_by_character(character)
        character_docs[character] = " ".join(comments)
    return character_docs


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
        rankings.append({
            "id": ids[i],
            "score": similarities[i],
            "text": docs[i]
        })
    
    return rankings



# Helper function: match if there is less than k edit distance from some term
def fuzzy_term_match(query, document, k):
    query = query.lower()
    terms = document.lower().split()

    for term in terms:
        if edit_distance(query, term) < k:
            return 1
    return 0

    # problem: wouldn't work for names longer than a space



nltk.download('vader_lexicon')
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