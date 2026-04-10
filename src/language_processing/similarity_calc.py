# file for similarity calculations and similar helper functions
import os # TODO: SHOULD WE ADD THIS TO PIPINSTALL REQUIREMENTS?? -derek
import numpy as np
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime
from rapidfuzz.distance import Levenshtein # REMIDNER TO ADD RAPIDFUZZ TO PIPINSTALL REQUIREMENTS




# try referencing csv files by joining path names
current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where similarity_calc.py lives
rp_path = os.path.join(current_dir, "csv", "reverse_postings_alias_exact.csv") # get inverted index info
rp = pd.read_csv(rp_path)

pfc_path = os.path.join(current_dir, "data", "piratefolk_comments.csv")
pfc = pd.read_csv(pfc_path) # comments with ids, texts, and other fields
    # ohhh it's short for pirate folk comments


# dict mapping character name to list of aliases (translations, canon nicknames, etc.)
# aliases gathered from the one piece wiki
# does not cover cases of reddit-given nicknames
name_variants = {
    # Straw Hat Pirates
    "Monkey D. Luffy": ["Luffy", "Monkey D. Rufi", "Rufi", "Ruffy", "Monch D. Roof"],
    "Roronoa Zoro": ["Zoro", "Roronoa Zolo", "Zolo", "Suron"],
    "Nami": [],
    "Usopp": ["Usoppu", "Usop", "Liar Bo", "Crook Bo", "Swindle Bo"],
    "Sogeking": ["Soge King", "Sniper King"],
    "Sanji": ["Sangi", "Sunkist"],
    "Tony Tony Chopper": ["Chopper"],
    "Nico Robin": ["Robin", "Lobin", "Cat Lowbun"],
    "Franky": ["Flanky"],
    "Cutty Flam": ["Cutty Fran", "Kati Fram"],
    "Brook": ["Brooke"],
    "Jinbe": ["Jimbei", "Jinbei", "Jimbe"],

    # Straw Hat Allies & Recurring Characters
    "Nefertari Vivi": ["Vivi", "Nefeltari Vivi", "Nefertari Bibi", "Vivi Nefertari"],
    "Nefertari Cobra": ["Cobra", "Nefeltari Cobra", "Nefeltari Nebra"],
    "Nefertari Titi": [],
    "Shanks": [],
    "Silvers Rayleigh": ["Rayleigh"],
    "Shakuyaku": ["Shakky"],
    "Portgas D. Ace": ["Ace", "Portgaz D. Ace", "Portgaz D. Trace"],
    "Sabo": [],
    "Monkey D. Garp": ["Garp"],
    "Monkey D. Dragon": ["Dragon"],
    "Koby": ["Coby", "Cobby"],
    "Helmeppo": [],
    "Makino": [],
    "Dadan": [],
    "Curly Dadan": [],
    "Woop Slap": [],
    "Genzo": [],
    "Nojiko": [],
    "Bell-mère": ["Belle-Mère", "Bellemere"],
    "Kaya": [],
    "Merry": [],
    "Iceburg": ["Iceberg", "Icebarg", "Icebarge"],
    "Paulie": ["Pauly"],
    "Kokoro": [],
    "Chimney": [],
    "Zambai": [],
    "Duval": [],
    "Hatchan": ["Hachi"],
    "Camie": ["Keimi", "Caymy"],
    "Pappag": ["Pappagu", "Pappug"],
    "Karoo": ["Carue", "Kalu", "Karu"],
    "Cavendish": [],
    "Bartolomeo": [],
    "Sai": [],
    "Leo": [],
    "Kyros": [],
    "Rebecca": [],
    "Viola": [],
    "Riku Doldo III": ["Riku"],
    "Trafalgar D. Water Law": ["Law", "Trafalgar Law"],
    "Bepo": [],
    "Kin'emon": [],
    "Momonosuke": [],
    "Kanjuro": [],
    "Raizo": [],
    "Denjiro": [],
    "Kawamatsu": [],
    "Ashura Doji": [],
    "Inuarashi": [],
    "Nekomamushi": [],
    "Carrot": [],
    "Yamato": [],
    "Pedro": [],
    "Wanda": [],
    "Hiriluk": ["Dr. Hiluluk", "Dr. Hiruluk"],
    "Kureha": ["Dr. Kureha"],
    "Dalton": [],
    "Dorry": [],
    "Brogy": [],
    "Coribou": [],
    "Marigold": [],
    "Sandersonia": [],
    "Boa Hancock": ["Hancock"],
    "Boa Marigold": [],
    "Boa Sandersonia": [],
    "Marguerite": ["Margaret"],
    "Emporio Ivankov": ["Ivankov", "Emporio Ivancov", "Emporio Iwankov"],
    "Inazuma": [],
    "Bentham": ["Bon Clay", "Bon Kurei", "Von Creay"],
    "Crocodile": [],
    "Daz Bonez": [],
    "Emporio Ivankov": ["Ivankov", "Emporio Ivancov", "Emporio Iwankov"],
    "Jinbe": ["Jimbei", "Jinbei", "Jimbe"],
    "Koala": [],
    "Hack": [],
    "Otohime": [],
    "Neptune": [],
    "Fukaboshi": [],
    "Ryuboshi": [],
    "Manboshi": [],
    "Shirahoshi": [],
    "Jaguar D. Saul": ["Saul", "Hagawa D. Saulo", "Jaguar D. Saulo", "Hagwarl D. Saulo", "Hagwar D. Sauro"],
    "Mont Blanc Cricket": ["Cricket", "Montblanc Cricket", "Monbran Cricket", "Mombran Cricket"],
    "Mont Blanc Noland": ["Noland", "Montblanc Noland", "Monbran Noland", "Mombran Noland", "Montblanc Norland"],
    "Laki": [],
    "Wyper": ["Wyler", "Wiper", "Waipa"],
    "Kalgara": ["Calgara"],
    "Enel": ["Eneru", "Ener"],
    "Wiper": [],

    # Marines & World Government
    "Sengoku": ["Zango"],
    "Tsuru": ["Crane"],
    "Smoker": ["Chaser"],
    "Tashigi": [],
    "Fullbody": ["Finbudi"],
    "Jango": ["Django"],
    "Hina": [],
    "Aokiji": ["Kuzan"],
    "Akainu": ["Sakazuki"],
    "Kizaru": ["Borsalino"],
    "Fujitora": ["Issho"],
    "Ryokugyu": ["Aramaki"],
    "Sengoku": ["Zango"],
    "Garp": [],
    "Coby": [],
    "Brannew": [],
    "Hannyabal": ["Hannibal", "Hannybal"],
    "Magellan": [],
    "Shiryu": ["Shilliew"],
    "Imu": ["Im"],
    "Sterry": ["Stelly"],
    "Wapol": [],
    "Nefertari Cobra": ["Cobra", "Nefeltari Cobra", "Nefeltari Nebra"],
    "Cp9": [],
    "Rob Lucci": ["Lucci", "Rob Rucchi"],
    "Kaku": [],
    "Kalifa": ["Califa"],
    "Jabra": ["Jyabura", "Jabura"],
    "Fukurou": ["Fukurô"],
    "Kumadori": [],
    "Spandam": [],
    "Blueno": [],
    "Bartholomew Kuma": ["Kuma", "Bisoromi Bear"],

    # Warlords / Shichibukai
    "Dracule Mihawk": ["Mihawk", "Juraquille Mihawk", "Mihark"],
    "Gecko Moria": ["Moria", "Gekko Moriah"],
    "Donquixote Doflamingo": [
        "Doflamingo",
        "Don Quixote Doflamingo",
        "Tanjiahdo Lofulamingo",
        "Don Quichotte Doflamingo",
        "Don Quichotte de Flamingo",
    ],
    "Donquixote Rosinante": ["Rosinante", "Corazon", "Don Quixote Rocinante", "Donquixote Rocinante", "Don Quichotte Rocinante"],
    "Buggy": ["Parchy"],
    "Perona": ["Perhona"],
    "Absalom": [],

    # Yonko & Crews
    "Edward Newgate": ["Newgate", "Whitebeard", "Shirohige", "Edward Newcart", "White Facial Hair"],
    "Marco": [],
    "Portgas D. Ace": ["Ace", "Portgaz D. Ace", "Portgaz D. Trace"],
    "Jozu": ["Joz", "Jose", "Jaws"],
    "Vista": [],
    "Thatch": [],
    "Squard": ["Squardo", "Squad", "Squado"],
    "Kaidou": ["Kaido"],
    "King": [],
    "Queen": [],
    "Jack": [],
    "Yamato": [],
    "Charlotte Linlin": ["Linlin", "Charlotte Rinrin", "Charlotte Lingling", "Big Mom", "Big Mam"],
    "Charlotte Katakuri": ["Katakuri", "Charlotte Dogtooth"],
    "Charlotte Smoothie": ["Smoothie"],
    "Charlotte Cracker": ["Cracker"],
    "Charlotte Pudding": ["Pudding", "Charlotte Purin"],
    "Charlotte Basskarte": ["Basskarte", "Charlotte Bassquarte"],
    "Shanks": [],
    "Lucky Roux": ["Lucky Roo"],
    "Benn Beckman": [],
    "Yasopp": [],
    "Marshall D. Teach": [
        "Teach",
        "Blackbeard",
        "Kurohige",
        "Marshall D. Teech",
        "Masuru D. Chocheh",
        "Black Facial Hair",
    ],
    "Jesus Burgess": ["Burgess", "G. Zass Burgess", "Xusasu Basasu"],
    "Van Augur": ["Augur", "Cloud Ouga", "Van Auger", "Van Ogre"],
    "Doc Q": ["Doku Q"],
    "Laffitte": ["Lafitte", "Raffit", "Lafeita"],
    "Catarina Devon": ["Devon", "Catalina Devon"],
    "Shiryu": ["Shilliew"],
    "Rocks D. Xebec": ["Rocks", "Xebec", "Rox"],

    # Villains & Antagonists
    "Arlong": [],
    "Hody Jones": ["Hody", "Hordy Jones", "Hodi Jones"],
    "Crocodile": [],
    "Nico Robin": ["Robin", "Lobin", "Cat Lowbun"],
    "Mr. 1": [],
    "Mr. 3": [],
    "Mr. 4": [],
    "Mr. 5": [],
    "Baroque Works": [],
    "Caesar Clown": ["Caesar", "Caesar Crown"],
    "Vergo": [],
    "Monet": ["Mone"],
    "Trebol": ["Trevor"],
    "Diamante": [],
    "Pica": [],
    "Sugar": [],
    "Giolla": ["Jora"],
    "Baby 5": [],
    "Buffalo": [],
    "Senor Pink": [],
    "Machvise": [],
    "Lao G": [],
    "Dellinger": [],
    "Bellamy": ["Binami"],
    "Enel": ["Eneru", "Ener"],
    "Ohm": ["Aum", "Orm", "Ohmu", "Om"],
    "Satori": [],
    "Shura": [],
    "Gedatsu": [],
    "Wapol": [],
    "Chess": [],
    "Kuromarimo": [],
    "Don Krieg": [],
    "Gin": [],
    "Pearl": [],
    "Kuro": [],
    "Sham": ["Siam"],
    "Buchi": ["Butchie"],
    "Porchemy": ["Polchemy", "Polchemi"],
    "Alvida": [],
    "Buggy": ["Parchy"],
    "Cabaji": [],
    "Mohji": [],
    "Richie": [],
    "Enishida": ["Genista"],
    "Oars": ["Oz", "Odr", "Odz", "Ohz"],
    "Oars Jr.": [],
    "Ryuma": ["Ryuuma", "Ryouma"],
    "Hogback": [],
    "Absalom": [],
    "Shyarly": ["Sharley", "Shirley"],
    "Wadatsumi": [],
    "Big Pan": ["Big Bun"],
    "Eustass Kid": ["Kid", "Eustass Kidd"],
    "Killer": [],
    "Apoo": [],
    "Hawkins": [],
    "Drake": [],
    "Urouge": [],
    "Capone Bege": ["Bege"],
    "Chew": ["Choo", "Chuu"],
    "Kuroobi": [],
    "Nami": [],

    # Other Notable Characters
    "Going Merry": ["Merry Go"],
    "Thousand Sunny": [],
    "Aladine": ["Aladdin"],
    "Aremo Ganmi": ["Peekatha Krotch"],
    "Banchi": ["Bunchi"],
    "Bartholomew Kuma": ["Kuma", "Bisoromi Bear"],
    "Chouchou": ["Shushu"],
    "Clou D. Clover": ["Clover", "Claiomh D. Clover"],
    "Corto": ["Colt"],
    "Fukurou": ["Fukurô"],
    "Gatherine": ["Gyatharin"],
    "Grabar": ["Grabba"],
    "Hasami": ["Scissors", "Pincers"],
    "Holy": ["Holly"],
    "Ikaros Much": ["Icaros Muhhi"],
    "Kumashi": ["Kumacy", "Kuma-C"],
    "Matsuge": ["Eyelashes", "Eyelash", "Lashes"],
    "Ochoku": ["Wang Zhi"],
    "Stronger": ["Strongheart"],
    "Su": ["Suu"],
    "Tibany": ["Elizabeth"],
    "Sterry": ["Stelly"],
    "Shelly": ["Sherry"],
    "Koza": ["Kohza"],
}

# input: dict mapping official name to list of aliases
# output: dict mapping official name to list of alises including official name
# context: one-time helper to make life easier for query_character
def make_name_variants_inclusive(name_variants):
    result = {}
    for official, alias_list in name_variants.items():
        inclusive_list = alias_list.copy()
        inclusive_list.insert(0, official)
        result[official] = inclusive_list
    return result

name_variants_inclusive = make_name_variants_inclusive(name_variants) 

# convert inclusive name_variants to list of aliases
# flatten list - credit to stackoverflow
list_all_aliases_nested = list(name_variants_inclusive.values())
list_all_aliases = [
    alias
    for nested_list in list_all_aliases_nested
    for alias in nested_list
]



# !!! function copied from character_counts.py
# fuzzy match query against all character names and aliases, return canonical name
# intending to be used in routes.py
# currently ununsed. not sure if we'll ever use this function.
def fuzzy_match_character(query: str, names_and_variants: dict[str, list[str]], threshold=1) -> str:
    best_match = None
    best_distance = float('inf')
    query_lower = query.lower()
    
    for char, aliases in names_and_variants.items():
        all_names = [char] + aliases
        for name in all_names:
            distance = Levenshtein.distance(query_lower, name.lower())
            if distance < best_distance:
                best_distance = distance
                best_match = char

    if best_distance <= threshold:
        return best_match
    return ""

















# input: character name
# output: list of comment ids that mention that character
def get_comments_by_character(character: str) -> list[str]:
    row = rp[rp["character"] == character]
    if row.empty:
        return []
    id_string = row.iloc[0]["comment_ids"]
    ids = id_string.split(",")
    
    # get matching comments
    comments = pfc[pfc["id"].isin(ids)]["text"].tolist()
    
    return comments

# outputs a dict that maps each character name to a string concatenating all comments
# that mention that character.
def build_character_docs() -> dict[str, str]:
    character_docs = {}
    for character in rp["character"]:
        comments = get_comments_by_character(character)
        character_docs[character] = " ".join(comments)
    return character_docs

# input: dict mapping character names to the concatenation of comments mentioning that name
# output:
#        - list of characer names
#        - tfidf vectorizer
#        - tfidf matrix:
#                - rows correspond to characters
#                - columns correspond to terms
def create_character_tfidf(character_docs: dict[str, str]):
    characters = list(character_docs.keys())
    docs = list(character_docs.values())

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    return characters, vectorizer, tfidf_matrix

character_docs = build_character_docs()
characters, vectorizer, tfidf_matrix = create_character_tfidf(character_docs)





# input:
#       - query string
#       - tfidf vectorizer
#       - tfidf matrix
#       - list of character names corresponding to rows of tfidf matrix
# output: charater name with highest cosine sim to query
#       - if query is an exact match for a character name, then just return that name
def query_character(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, characters: list[str], top_k: int = 1) -> str:
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_index = sims.argmax()

    query = query.lower()
    alias_list_lower = [alias.lower() for alias in list_all_aliases]
    if query in alias_list_lower:
        return query
    else:
        return characters[best_index]
    # TODO: code doesn't use top_k yet

def make_pickle():
    joblib.dump({
    "matrix": tfidf_matrix,
    "vectorizer": vectorizer,
    "characters": characters
}, "data/model.pkl")
    
# make_pickle()






# 2026-04-05 ADDING FUNCTIONS FROM A PREVIOUS PROTOYPE
# in order to retrieve comments actually relevant to the query
# rather than comments similar to character


# helper for retrieve_k_sim_comments
# given a csv of comment data, create a tf-idf matrix with
#       rows: comments
#       cols: terms
def create_comment_term_tfidf_matrix(filepath: str):
    df = pd.read_csv(filepath)

    df["text"] = df["text"].fillna("").astype(str)

    texts = df["text"].tolist() # list of comment "bodies"
    ids = df["id"].tolist() # list of comment ids

    comment_term_vectorizer = TfidfVectorizer()
    comment_term_tfidf_matrix = comment_term_vectorizer.fit_transform(texts)
    return (ids, comment_term_vectorizer, comment_term_tfidf_matrix, texts)



# return k most relevant documents for query
# 	- input example: "Most useless character in Whole Cake arc"
# 	- output: Ranked comments based off decreasing cosine similarity
#               list of dicts, each representing a comment
def retrieve_k_sim_comments(query, vectorizer, comment_term_tfidf_matrix, ids, texts, k):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, comment_term_tfidf_matrix).flatten()
        # list of cosine similarities for each comment

    # get just top k indeces
    top_indices = similarities.argsort()[::-1][:k]

    rankings = []
    for i in top_indices:
        rankings.append((ids[i], similarities[i]))
    
    return rankings



# Code below is for creating the comment_term_tfidf_matrix, to be used for retrieving relevant comments
print("\033[92m Start create_comment_term_tfidf_matrix \033[0m")
(comment_ids, comment_term_vectorizer, comment_term_tfidf_matrix, texts) = \
create_comment_term_tfidf_matrix(pfc_path)
print("\033[92m End create_comment_term_tfidf_matrix \033[0m")










# TESTS ----------------------------------------------------------------------------------------
# print("below is case sensitive teest")
# print(get_comments_by_character("Kuro") == get_comments_by_character("kuro"))
# print("below is the query test")
# print(query_character("Akainu", vectorizer, tfidf_matrix, characters))
# print("enies lobby?")
# print([c for c in get_comments_by_character("akainu") if "him" in c.lower()])


