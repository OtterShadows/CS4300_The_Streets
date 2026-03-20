import pandas as pd
import spacy
from collections import Counter
import csv

#first function get counts of characters output to  text file
nlp = spacy.load("en_core_web_sm")
docs = pd.read_csv("data/piratefolk_comments.csv")
comments = docs["text"].dropna().tolist()

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
    reverse_postings = {}

    texts = docs["text"].astype(str)
    ids = docs["id"]

    for doc, comment_id in zip(nlp.pipe(texts, batch_size = 1000), ids):
        persons = set(ent.text for ent in doc.ents if ent.label_ == "PERSON")

        for person in persons:
            reverse_postings[person].append(comment_id)

    return reverse_postings


char_count = charCount()
print(char_count)
write_char_counts_to_csv(char_count, "data/character_names.csv")

print(createReversePostings())


    
    


