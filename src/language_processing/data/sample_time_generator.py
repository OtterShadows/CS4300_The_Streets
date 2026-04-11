import requests
import os

# script for retrieving a bunch of json files of comments from the piratefolk subreddit

# stores in batches of 100 comments, one file for each week, in new folder "comments_data"

# uses pushpull.io api


current_unix_timestamp = 1775889722 # for Apr 11, 2026
day = 24 * 60 * 60
week = 7 * day
year = 365 * day
year_2023 = 1672531200 # jan 1, 2023
year_2024 = year_2023 + year
year_2025 = year_2023 + year * 2
year_2026 = year_2023 + year * 3


# end_timestap is the later time, and start_timestamp is the earlier time.
def list_timestamps(start_timestamp, end_timestamp, interval_seconds):
    timestamps = []
    current_timestamp = start_timestamp
    while current_timestamp <= end_timestamp:
        timestamps.append(current_timestamp)
        current_timestamp += interval_seconds
    return timestamps



def get_jsons(list_timestamps):
    if len(list_timestamps) == 0:
        return
    elif len(list_timestamps) > 1000:
        print("Too many timestamps.")
        return
    # print(list_timestamps)

    i = 1
    for timestamp in list_timestamps:
        link = f"https://api.pullpush.io/reddit/search/comment/?subreddit=piratefolk&size=100&before={timestamp}"
        link_content = requests.get(link).content
        path = "src/language_processing/data/new_comments_data/2025"
        json = open(f'src/language_processing/data/new_comments_data/2025/comments_{timestamp}.json', 'wb').write(link_content)
        print(f"Retrieved json for timestamp {timestamp}, number {i}/{len(list_timestamps)}")
        i = i + 1
    
    return



# start = current_unix_timestamp - (4 * year)
# end = current_unix_timestamp
start = year_2025
end = year_2026 - 1
get_jsons(list_timestamps(start, end, week / 2))


