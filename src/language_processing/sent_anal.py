

#load sentiment analyis model
#categorize as positive negative neutral
#sentiments are an enum of positive, negative, neutral
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import character_counts

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')
    
sia = SentimentIntensityAnalyzer()

#if a character has a slander/glaze name, it automatically becomes positive or negative. e.g Zoro Woro-> positive, Loro-> negative
def slander_glaze_sentiment(names_and_variants, text):
    vowels = set("aeiouAEIOU")
    for char, aliases in names_and_variants.items():
        all_names = [char] + aliases
        for name in all_names:
            if "l"+name[1:] in text:
                return "negative"
            if "w" + name[1:] in text:
                return "positive"
def get_sentiment(text):
    # Check for slander/glaze sentiment first
    glaze_sentiment = slander_glaze_sentiment(character_counts.names_and_variants, text)
    if glaze_sentiment:
        return glaze_sentiment
    score = sia.polarity_scores(text)["compound"]
    
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    else:
        return "neutral"

"""print(get_sentiment("I love this game!")) # positive
print(get_sentiment("This game is terrible.")) # negative
print(get_sentiment("This game is okay.")) # neutral"""
