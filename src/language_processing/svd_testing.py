from scipy.sparse.linalg import svds
import os
import joblib
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# matplotlib.use("TkAgg")

#   quick fix for dependency issue
#   may not be the right move, i don't know the code in this file well -DT

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where svd_testing.py lives, language_processing
model_path = os.path.join(current_dir, "data", "model.pkl")
data = joblib.load(model_path)
vectorizer = data["vectorizer"]
td_matrix = data["matrix"]
characters =data["characters"]
#u, s, v_trans = svds(td_matrix, k=100)
print(f"DEBUG: Loaded model.pkl with {len(characters)} characters and td_matrix shape {td_matrix.shape}")
print(f"DEBUG: Sample characters: {characters[:5]}")
print(f"DEBUG: Sample td_matrix row: {td_matrix[0].toarray()[:5]}")


reverse_postings_filename = "reverse_postings_(well).csv"
rp_path = os.path.join(current_dir, "csv", reverse_postings_filename) # get inverted index info
rp = pd.read_csv(rp_path)

piratefolk_comments_filename = "piratefolk_comments_(v2).csv"
pfc_path = os.path.join(current_dir, "csv", piratefolk_comments_filename) #
pfc = pd.read_csv(pfc_path)

"""      print(td_matrix.shape)
print(u.shape)
print(s.shape)
print(v_trans.shape)"""

"""
plt.plot(s[::-1])
plt.xlabel("Singular value number")
plt.ylabel("Singular value")
plt.show()"""

docs_compressed, s, words_compressed = svds(td_matrix, k=15)
words_compressed = words_compressed.transpose()
word_to_index = vectorizer.vocabulary_       
index_to_word = {i:t for t,i in word_to_index.items()}
words_compressed_normed = normalize(words_compressed, axis = 1)
docs_compressed_normed = normalize(docs_compressed)

print(f"DEBUG: docs_compressed_normed shape: {docs_compressed_normed.shape}, words_compressed_normed shape: {words_compressed_normed.shape}")

def make_pickle():
    joblib.dump({
    "svd_words_compressed": words_compressed_normed,
    "s": s,
    "svd_docs_compressed": docs_compressed_normed
}, "src/language_processing/data/svd_model.pkl")
# make_pickle()

def closest_docs_to_query(query):
    print(f"DEBUG: Characters from model.pkl: {characters}")
    k=5
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    return [characters[i] for i in asort[1:]]

def closest_doc_to_query(query):
    print(f"DEBUG: Characters from model.pkl: {characters}")
    k=5
    query_tfidf = vectorizer.transform([query]).toarray()
    query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
    sims = docs_compressed_normed.dot(query_vec)
    asort = np.argsort(-sims)[:k+1]
    return characters[asort[1]]
# print(closest_doc_to_query("potential man"))
            
"""for i, proj, sim in closest_docs_to_query(query_vec):
    doc_svd_vec = docs_compressed_normed[i]
    top_dims_indices = np.argsort(np.abs(doc_svd_vec))[::-1][:3]
    top_dims = [int(dim) for dim in top_dims_indices]
    print("({}, {}, {:.4f}, Top dimensions: {}) ".format(i, proj, sim, top_dims))"""

names_and_variants = {
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
    "Koby": ["Coby"],
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
    "Kin'emon": ["Kinemon"],
    "Momonosuke": [],
    "Kanjuro": [],
    "Raizo": [],
    "Denjiro": [],
    "Kawamatsu": [],
    "Ashura Doji": [],
    "Inuarashi": [],
    "Nekomamushi": ["Neko"],
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
    "Loki": [],
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
    "Brannew": [],
    "Hannyabal": ["Hannibal", "Hannybal"],
    "Magellan": [],
    "Shiryu": ["Shilliew"],
    "Imu": [],
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
    "Eustass Kid": ["Kid", "Eustass Kidd", "Useless Captain Mid","Captain Mid"], 
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

    

#retrieve top words for each dimension in the svd matrix

def top_words_for_dimension(words_compressed_in):
    all_names = []
    for name, variants in names_and_variants.items():
        all_names.append(name.lower())
        all_names.extend([v.lower() for v in variants])
    for i in range(15):
        print("Top words in dimension", i)
        dimension_col = words_compressed[:,i].squeeze()
        asort = np.argsort(-dimension_col)
        #remove characternames from the top words, since they are not meaningful for understanding the dimension.
        wlist = []
        w = 0
        while wlist.__len__() < 15:
            if index_to_word[asort[w]] not in all_names:
                wlist.append(index_to_word[asort[w]])
            w+=1
        print(wlist)
        print()

# context: appropriate the SVD model made for character docs for retrieval/ranking of comments,
# hopefully ones that match the meaning of the query better...
# for use in routes.py
# output: list of tuples of form (comment_text, sim_score)
def svd_retrieve_k_sim_comments(character: str, query: str, tfidf_matrix, k = 20):
    # get all comments mentioning character from reverse postings
    print(f"\033[95mUsing SVD retrieval for character '{character}' and query '{query}'\033[0m")
    row = rp[rp["character"] == character]
    if row.empty:
        return []

    ids_string = row.iloc[0]["comment_ids"] # comma separated string of ids
    ids_list = ids_string.split(",")

    matched_df = pfc[pfc["id"].isin(ids_list)].copy()
        # filter to only have ids in ids_list
    if matched_df.empty:
        return []

    indices = row.index.tolist()
    comment_tfidf_matrix = vectorizer.transform(matched_df["text"].fillna("")).toarray()
    comments_compressed = normalize(comment_tfidf_matrix.dot(words_compressed))

    query_vec = vectorizer.transform([query])
    query_vec_comp = normalize(query_vec.toarray().dot(words_compressed)).squeeze()
    sims = comments_compressed.dot(query_vec_comp)
    ranked = np.argsort(-sims)[:k]
    print(f"DEBUG: In retrieval function, len(ranked) = {len(ranked)} for character '{character}' and query '{query}'")

    result_tuples = [(matched_df.iloc[i]["id"], float(sims[i])) for i in ranked]
    return result_tuples





# UNUSED? -------------------------------------------------------------------------------
def closest_words(word_in, words_representation_in, k = 10):
    if word_in not in word_to_index: return "Not in vocab."
    sims = words_representation_in.dot(words_representation_in[word_to_index[word_in],:])
    asort = np.argsort(-sims)[:k+1]
    return [(index_to_word[i],sims[i]) for i in asort[1:]]



# TEST -----------------------------------------------------------------------------------

# query =" potential man"
# query_tfidf = vectorizer.transform([query]).toarray()
# query_vec = normalize(query_tfidf.dot(words_compressed)).squeeze()
#top_words_for_dimension(words_compressed)
#make_pickle()