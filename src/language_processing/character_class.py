import pandas as pd
import sent_anal
from datetime import datetime
import joblib 


comments_df = pd.read_csv("data/piratefolk_comments.csv")
comments_df = comments_df.set_index("id")

postings_df = pd.read_csv("src/language_processing/reverse_postings.csv")
postings_df = postings_df.drop_duplicates(subset="character")
postings_df = postings_df.set_index("character")

class Rating:
    def __init__(self, date, rating, sentiment):
        self.date = date
        self.rating = rating
        #sent anal positive, negative, neutral
        assert sentiment in ["positive", "negative", "neutral"]
        self.sentiment = sentiment


class Comment:
   def __init__(self, user, text, sentiment, rating=None, score=None, timestamp=None, controversiality=None):
        self.user = user
        self.text = text
        self.sentiment = sentiment
        self.rating = rating
        self.score = score
        self.timestamp = timestamp
        self.controversiality = controversiality

        # comment 
# id,timestamp,score,controversiality,text


#creates comment object using comment id, will most likely be used in a loop iterating through reverse postings
def get_comment(id):
    row = comments_df.loc[id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    text = str(row['text'])
    sentiment = sent_anal.get_sentiment(text)
    score = float(row['score'])
    timestamp = float(row['timestamp'])
    controversiality = int(row['controversiality'])

    return Comment(
        user='Pirate_Man22',
        text=text,
        sentiment=sentiment,
        rating=4.5,
        score=score,
        timestamp=timestamp,
        controversiality=controversiality
    )

#uses character name to create the rating over time using character name
def get_rating_over_time(charName):
    #get all comments for the character, then create a list of ratings over time
    ids = postings_df.loc[charName, "comment_ids"]
    if isinstance(ids, pd.Series):
        ids = ids.iloc[0]
    comments = ids.split(",")
    #make list of comment objects using get_comment function then sort by timestamp
    comments = sorted([get_comment(comment_id) for comment_id in comments], key=lambda x: x.timestamp)
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
        print(f"Date: {date}, Sentiment: {sentiment}, Rating: {rating}")
        ratings_over_time.append(Rating(date, rating, sentiment))
    return sorted(ratings_over_time, key=lambda x: x.date)
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
        self.total_comments = len(comments)
        #ask gabby what the difference was supposed to be... might be the ranked most relevant comments?
        self.retrieved = comments if comments is not None else[]
    

def create_character(name):    
    comments = postings_df.loc[name, "comment_ids"].split(",")
    comment_list = []
    for comment in comments:
        comment_list.append(get_comment(comment))
    retrieved = comment_list[:5]
    ratings_over_time = get_rating_over_time(name)
    summary = "This is a summary of the character."
    sentiment_score = ratings_over_time[-1].rating
    sentiment = comment_list[-1].sentiment
    total_comments = len(comment_list)
    if sentiment_score > 100:
        rank = "A"
    elif sentiment_score < 80:
        rank = "C"
    else:        
        rank = "B"
    print(f"Character: {name}, Rank: {rank}, Total Comments: {total_comments}, Sentiment: {sentiment}, Sentiment Score: {sentiment_score}")
    return Character(name, rank, total_comments, sentiment, sentiment_score, summary, ratings_over_time, comment_list, retrieved)
#create_character("Jika")

def create_all_characters():
    characters = []
    for charName in postings_df.index:
        character = create_character(charName)
        characters.append(character)
    return characters

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
            "ratings_over_time": [{"date": r.date.timestamp(), "rating": r.rating, "sentiment": r.sentiment} for r in character.ratings_over_time],
            #comments as a list of dicts
            "comments": [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality} for c in character.comments],
            #retrieved as a list of dicts
            "retrieved": [{"user": c.user, "text": c.text, "sentiment": c.sentiment, "rating": c.rating, "score": c.score, "timestamp": c.timestamp, "controversiality": c.controversiality} for c in character.retrieved]
        }
    return char_dict

#print(create_all_characters())
joblib.dump(characters_to_dict(create_all_characters()), "data/character_data.pkl")


