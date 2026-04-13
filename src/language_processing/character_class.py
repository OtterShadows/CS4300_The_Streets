import os
import pandas as pd
from language_processing import sent_anal
from datetime import datetime
import re
import joblib 
# from src.language_processing import similarity_calc

#To speed up multiple calls of the functions.
comment_cache = {}
# comments is a csv with columns id, timestamp, score, controversiality, text
# comments_df = pd.read_csv("data/piratefolk_comments.csv")
comments_df_path = os.path.join(current_dir, "csv", "new_pf_comments.csv")
comments_df = pd.read_csv(comments_df_path) 
comments_df = comments_df.set_index("id")

# postings is a csv with columns character, comment_ids (comma separated)
# postings_df_path = pd.read_csv("src/language_processing/csv/reverse_postings_alias_exact.csv")
def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    comments_df = pd.read_csv(
        os.path.join(current_dir, "data", "piratefolk_comments.csv")
    ).set_index("id")

    def is_valid(text):
        words = re.findall(r'\b\w+\b', str(text))
        return len(words) > 1

    comments_df = comments_df[comments_df["text"].apply(is_valid)]

    postings_df = pd.read_csv(
        os.path.join(current_dir, "csv", "reverse_postings_alias_exact.csv")
    ).drop_duplicates(subset="character").set_index("character")

    valid_ids = set(comments_df.index)

    postings_df["comment_ids"] = postings_df["comment_ids"].apply(
        lambda x: ",".join(cid for cid in x.split(",") if cid in valid_ids)
    )

    return comments_df, postings_df

class Rating:
    def __init__(self, date, rating, sentiment):
        self.date = date
        self.rating = rating
        #sent anal positive, negative, neutral
        assert sentiment in ["positive", "negative", "neutral"]
        self.sentiment = sentiment


class Comment:
   def __init__(self, user, text, sentiment, rating=None, score=None, timestamp=None, controversiality=None, sim_score=None):
        self.user = user
        self.text = text
        self.sentiment = sentiment
        #do we have this elsewhere when you set get comment
        self.rating = rating
        self.score = score
        self.timestamp = timestamp
        self.controversiality = controversiality
        self.sim_score = sim_score


# new version of get_comment to account for similarity score
def create_comment(id, sim_score, comments_df):
    if id in comment_cache:
        return comment_cache[id]
    if id not in comments_df.index:
        return None
    row = comments_df.loc[id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    text = str(row["text"])
    sentiment = sent_anal.get_sentiment(text)
    comment = Comment(
        user=None,
        text=text,
        sentiment=sentiment,
        rating=0,
        score=float(row["score"]),
        sim_score=sim_score,
        timestamp=float(row["timestamp"]),
        controversiality=int(row["controversiality"])
    )

    comment_cache[id] = comment
    return comment
    

#uses character name to create the rating over time using character name
def get_rating_over_time(charName, postings_df, comments_df):
    #get all comments for the character, then create a list of ratings over time
    ids = postings_df.loc[charName, "comment_ids"]

    if not isinstance(ids, str) or not ids.strip():
        return []
    comment_ids = [cid for cid in ids.split(",") if cid.strip()]
    
    #make list of comment objects using get_comment function then sort by timestamp
    comments = sorted(
    [
        c for c in (create_comment(cid, 0, comments_df) for cid in comment_ids)
        if c is not None
    ],
    key=lambda x: x.timestamp)
    ratings_over_time = []
    init_score = 100
    for comment in comments:
        date = datetime.fromtimestamp(comment.timestamp)
        if comment.sentiment == "positive":
            init_score += 20
        elif comment.sentiment == "negative":
            init_score -= 20
        #neutral does not change the score
        rating = init_score
        sentiment = comment.sentiment
        # print(f"Date: {date}, Sentiment: {sentiment}, Rating: {rating}")
        # ratings_over_time.append(Rating(date, rating, sentiment))
        ratings_over_time.append(Rating(datetime.fromtimestamp(comment.timestamp), init_score, comment.sentiment))
    return sorted(ratings_over_time, key=lambda x: x.date)
        #print(f"Date: {date}, Sentiment: {sentiment}, Rating: {rating}")

#get_rating_over_time("Jika")

class Character:
    def __init__(self, name, rank,total_comments, sentiment, sentiment_score, summary,
                 ratings_over_time=None, comments=None, retrieved=None):
        self.name = name
        #some trivial pattern matching function
        self.rank = rank
        #the sentiment that has the most in ratings over time
        self.sentiment = sentiment
        #make some metric using the enum
        self.sentiment_score = sentiment_score
        #ong put dummy data here for now
        self.summary = summary
        #complete ratings_over_time function
        self.ratings_over_time = ratings_over_time if ratings_over_time is not None else []
        #should be rating of final rating in ratings over time
        self.current_rating = self.ratings_over_time[len(self.ratings_over_time)-1].rating if self.ratings_over_time else 0
        self.comments = comments if comments is not None else []
        self.total_comments = total_comments
        #ask gabby what the difference was supposed to be... might be the ranked most relevant comments?
        self.retrieved = retrieved if retrieved is not None else[]
    

def create_character(name, postings_df, comments_df):
   # list of comment ids mentioning [name]  
    ids = postings_df.loc[name, "comment_ids"]

    if isinstance(ids, pd.Series):
        ids = ids.iloc[0]

    if not isinstance(ids, str) or not ids.strip():
        comment_ids = []
    else:
        comment_ids = [cid for cid in ids.split(",") if cid.strip()]

    comment_list = [ c for c in
        (create_comment(cid, 0, comments_df) for cid in comment_ids) if c is not None
    ]
    ratings_over_time = get_rating_over_time(name, postings_df, comments_df)
    pos = sum(1 for c in comment_list if c.sentiment == "positive")
    neg = sum(1 for c in comment_list if c.sentiment == "negative")

    if pos > neg:
        summary = f"There is an overall positive sentiment with {pos} positive vs {neg} negative comments."
    elif neg > pos:
        summary = f"There is an overall negative sentiment with {neg} negative vs {pos} positive comments."
    else:
        summary = "We found that there is mixed sentiment from the community."

    sentiment_score = ratings_over_time[-1].rating if ratings_over_time else 100
    sentiment = comment_list[-1].sentiment if comment_list else "neutral"

    rank = (
        "A" if sentiment_score > 100
        else "C" if sentiment_score < 80
        else "B"
    )

    return Character(
        name, rank, len(comment_list),
        sentiment, sentiment_score,
        summary,
        ratings_over_time,
        comment_list,
        comment_list[:5]
    )
#create_character("Jika")


def create_all_characters(postings_df, comments_df):
    return[create_character(name, postings_df, comments_df) for name in postings_df.index]

def characters_to_dict(characters):
    char_dict = {}
    for character in characters:
        char_dict[character.name] = {
            "rank": character.rank,
            "total_comments": character.total_comments,
            "sentiment": character.sentiment,
            "currentRating": character.sentiment_score,
            "summary": character.summary,
            #ratings as a list of dicts
            # "ratings_over_time": [(r.date.timestamp(), r.rating) for r in character.ratings_over_time],
            "ratings_over_time": [{"date": r.date.timestamp(), "rating": r.rating, "sentiment": r.sentiment} for r in character.ratings_over_time],
            #comments as a list of dicts
            "comments": [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality} for c in character.comments],
            #retrieved as a list of dicts
            "retrieved": [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality} for c in character.retrieved]
        }
    return char_dict
#print(create_all_characters())
# comments_df, postings_df = load_data()
# joblib.dump(characters_to_dict(create_all_characters(postings_df, comments_df)), "src/language_processing/src/language_processing/data/character_data.pkl")


