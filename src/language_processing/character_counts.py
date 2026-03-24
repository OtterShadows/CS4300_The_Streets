import pandas as pd
import spacy
from collections import Counter
import csv
from collections import defaultdict

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
def write_reverse_postings_to_csv(filename="reverse_postings.csv"):
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

# Run it
write_reverse_postings_to_csv()
#write_counts_to_csv()


char_count = charCount()
print(char_count)
write_char_counts_to_csv(char_count, "data/character_names.csv")

print(createReversePostings())


    
    


