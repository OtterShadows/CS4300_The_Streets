import pandas as pd
import os
import numpy as np
import spacy
from collections import Counter
import csv
from collections import defaultdict
from rapidfuzz.distance import Levenshtein # REMIDNER TO ADD RAPIDFUZZ TO PIPINSTALL REQUIREMENTS

# DATA --------------------------------------------------------------------------------------

# dict mapping character name to list of aliases (translations, canon nicknames, etc.)
# aliases gathered from the one piece wiki
# does not cover cases of reddit-given nicknames
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


#first function get counts of characters output to  text file
load_nlp = True
if load_nlp:
    nlp = spacy.load("en_core_web_sm")

current_dir = os.path.dirname(os.path.abspath(__file__)) #the path where character_counts.py lives (language_processing)
comments_path = os.path.join(current_dir, "csv", "new_pf_comments.csv")
docs = pd.read_csv(comments_path)
comments = docs["text"].dropna().tolist()








# FUNCTIONS ---------------------------------------------------------------------------------------

# will deprecate... going with manual list of characters/aliases instead of NER
def charCount():
    character_counts = Counter()
    for doc in nlp.pipe(comments, batch_size=1000):
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                character_counts[ent.text] += 1
    return character_counts

def write_char_counts_to_csv(character_counts: Counter, output_path: str):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["character", "count"])
        for character, count in character_counts.items():
            writer.writerow([character, count])


#create dictionary of touples (text, list of characters in text, time)
#create reverse postings for sent anal output to text file

def createReversePostings():
    # docs = docs
    # comments = comments
    reverse_postings = defaultdict(list)

    texts = docs["text"].fillna("").astype(str)
    ids = docs["id"]

    for doc, comment_id in zip(nlp.pipe(texts, batch_size = 1000), ids):
        persons = set(ent.text for ent in doc.ents if ent.label_ == "PERSON")

        for person in persons:
            reverse_postings[person].append(comment_id)

    return reverse_postings

def write_counts_to_csv(filename="character_counts.csv"):
    counts = charCount()
    
    df = pd.DataFrame(counts.items(), columns=["character", "count"])
    df = df.sort_values(by="count", ascending=False)

    df.to_csv(filename, index=False)



# fuzzy match query against all character names and aliases, return canonical name
# intending to be used in routes.py
def fuzzy_match_character(query: str, names_and_variants: dict[str, list[str]], threshold=0.3) -> str:
    best_match = None
    best_distance = float('inf')
    query_lower = query.lower()
    for char, aliases in names_and_variants.items():
        for name in [char] + aliases:
            distance = Levenshtein.distance(query_lower, name.lower())
            normalized = distance / max(len(query_lower), len(name))
            if normalized < best_distance:
                best_distance = normalized
                best_match = char
    if best_distance <= threshold:
        return best_match
    return ""

#print(fuzzy_match_character("stocks are up", names_and_variants))

# returns true if edit distance is less than or equal to threshold
def fuzzy_edit_distance(source: str, target: str, threshold: int = 0):
    return Levenshtein.distance(source, target) <= threshold
    # using pre-implemented edit-distance from rapidfuzz because it ended up being like 
    # 50x times faster when running code



# convert names_and_variants dict to csv file
# input: dict mapping each character name to list of aliases
# output: csv with columns character, alias
def aliases_to_csv(names_and_variants: dict[str, list[str]], output_path="src/language_processing/csv/character_aliases.csv"):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["character", "alias"])
        for character, aliases in names_and_variants.items():
            print(f"character: {character}, aliases: {aliases}")
            for alias in aliases:
                writer.writerow([character, alias])



# counts the number of times each character (or their aliases) are mentioned in the comments, returns a Counter of character counts
# input: dict mapping character name to list of aliases
# output: Counter object mapping character name to count of mentions (including aliases)
def char_count_alias(names_and_variants: dict[str, list[str]]):
    char_counts = Counter()
    for comment in nlp.pipe(comments, batch_size=1000):
        for word in comment.text.split():
            for character, aliases in names_and_variants.items():
                if word == character or word in aliases:
                    char_counts[character] += 1
    return char_counts



def write_reverse_postings_to_csv(filename="src/language_processing/csv/reverse_postings_alias.csv"):
    reverse_postings = createReversePostings()
    
    # Convert to DataFrame
    df = pd.DataFrame(
        [(person, ids) for person, ids in reverse_postings.items()],
        columns=["character", "comment_ids"]
    )
    
    # Optionally convert list to string for CSV
    df["comment_ids"] = df["comment_ids"].apply(lambda x: ",".join(map(str, x)))
    
    # Write to CSV
    df.to_csv(filename, index=False, encoding="utf-8")



# (helper for create_reverse_postings_alias)
# input: dict mapping character name to list of aliases
# output: dict mapping alias to canonical character name (including mapping canonical name to itself)
def create_alias_to_canonical_dict(names_and_variants: dict[str, list[str]]) -> dict[str, str]:
    alias_to_canonical = {}
    for char, aliases in names_and_variants.items():
        alias_to_canonical[char.lower()] = char
        for alias in aliases:
            alias_to_canonical[alias.lower()] = char
    return alias_to_canonical



# to store alias -> canoncial name mapping
def alias_to_canonical_dict_to_csv(alias_to_canonical: dict[str, str], output_path="src/language_processing/csv/alias_to_canonical.csv"):
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["alias", "canonical"])
        for alias, canonical in alias_to_canonical.items():
            writer.writerow([alias, canonical])



# UNFINISHED?
# create inverted index mapping character names to comment ids where they (or some alias) are mentioned
# returns dict[str, list[str]]
def create_reverse_postings_alias(filename="src/language_processing/csv/reverse_postings_alias.csv"):
    reverse_postings = defaultdict(list)
    texts = docs["text"].fillna("").astype(str)
    ids = docs["id"]

    alias_to_canonical_map = create_alias_to_canonical_dict(names_and_variants)

    
    comment_number = 0 # counting comments processed for debugging
    num_comments = 8801 # from csv
    for comment, comment_id in zip(nlp.pipe(texts, batch_size=1000), ids):
        
        comment_text = comment.text if hasattr(comment, "text") else str(comment)
        print(f"Processing comment ID: {comment_id}, {comment_text[:30]}...")
        print(f"Comment number: {comment_number} out of {num_comments}") 
        words = [w.strip() for w in comment_text.lower().split() if w.strip()]
        matched_chars = set()
        for word in words:
            for name_lower, canonical in alias_to_canonical_map.items():
                if fuzzy_edit_distance(name_lower, word, threshold=0):
                    matched_chars.add(canonical)
        for char in matched_chars:
            reverse_postings[char].append(comment_id)
        
        comment_number = comment_number + 1

    return reverse_postings


def write_reverse_postings_alias_to_csv(reverse_postings, filename="src/language_processing/csv/reverse_postings_alias.csv"):
    # Convert to DataFrame
    df = pd.DataFrame(
        [(person, ids) for person, ids in reverse_postings.items()],
        columns=["character", "comment_ids"]
    )
    
    # Optionally convert list to string for CSV
    df["comment_ids"] = df["comment_ids"].apply(lambda x: ",".join(map(str, x)))
    
    # Write to CSV
    df.to_csv(filename, index=False, encoding="utf-8")

#create nicknames with the standard W or L slander format, e.g. "Zoro" -> [Loro, Woro]
def transform(name: str):
        vowels = set("aeiou")
        name = name.strip()
        if not name:
            return []

        first = name[0].lower()

        # vowel case → prefix l or w
        if first in vowels:
            return [f"l{name}", f"w{name}"]

        # consonant case → replace first letter with l or w
        return [f"l{name[1:]}", f"w{name[1:]}"]
def create_slander_nicknames(characters: list[str]) -> dict[str, list[str]]:
    updated = {}
    for char, aliases in names_and_variants.items():
        all_names = [char] + aliases
        new_aliases = set(aliases)

        for name in all_names:
            for nick in transform(name):
                new_aliases.add(nick)

        updated[char] = list(new_aliases)
    return updated

names_and_variants = create_slander_nicknames(names_and_variants)
#print(names_and_variants["Roronoa Zoro"])

# RUNNING FUNCTIONS --------------------------------------------------------------------------------------------------------

#char_to_count = char_count_alias(names_and_variants)
    # maps character name to # mentions (including aliases)

#write_char_counts_to_csv(char_to_count, "src/language_processing/csv/new_character_counts.csv")
    # writes above dict to csv

#aliases_to_csv(names_and_variants, output_path="src/language_processing/csv/name_to_aliases.csv")
    # creates csv mapping character to alias



#reverse_postings_alias = create_reverse_postings_alias(filename="src/language_processing/csv/new_reverse_postings.csv")
    # dict mapping character name to list of comment ids where character (or some alias) is mentioned

#write_reverse_postings_alias_to_csv(reverse_postings_alias, filename="src/language_processing/csv/new_reverse_postings.csv")
    # writes above dict to csv



#alias_to_canonical = create_alias_to_canonical_dict(names_and_variants)
    # dict mapping alias to canonical character name (including mapping canonical name to itself)

#alias_to_canonical_dict_to_csv(alias_to_canonical, output_path="src/language_processing/csv/alias_to_canonical.csv")
    # writes above dict to csv




