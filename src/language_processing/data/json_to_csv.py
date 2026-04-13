import json
import os
import csv
from pathlib import Path
import re
import pandas as pd


def json_to_csv1():
    with open("data_set.csv", "w") as f:
        directory_path = Path('./data/comments_data')
        for file_path in directory_path.iterdir():
            if file_path.is_file():
                print(file_path.name)
                # You can also read the file content directly
                content = file_path.read_text()
                json_content = json.loads(content)
                for i in range(100):
                    try:
                        json_content['data'][i]
                    except KeyError:
                        continue
                    comment_id = json_content['data'][i]['id']
                    comment_text = json_content['data'][i]['body']
                    cleaned_comment_text = comment_text.replace('\n', ' ').replace('\r', ' ').replace('"', '')
                    comment_timestamp = json_content['data'][i]['created_utc']

                    f.write(f'{comment_id},{comment_timestamp},\"{cleaned_comment_text}\"\n')
    
def json_to_csv2():
    with open("src/language_processing/csv/new_piratefolk_comments_2025.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'timestamp', 'permalink', 'score', 'controversiality', 'author', 'text'])

        directory_path = Path('src/language_processing/data/new_comments_data/2025')

        for file_path in directory_path.iterdir():
            if file_path.is_file():
                print(f"{file_path.name}")
                content = file_path.read_text()
                json_content = json.loads(content)
                if 'data' not in json_content:
                    print(file_path.name + " does not contain 'data' key. Skipped.")
                    continue

                for comment in json_content.get('data', []):
                    comment_id = comment.get('id')
                    comment_timestamp = comment.get('created_utc')
                    comment_permalink = comment.get('permalink')
                    comment_author = comment.get('author')
                    comment_text = comment.get('body', '')
                    comment_score = comment.get('score', 0)
                    comment_controversiality = comment.get('controversiality', 0)

                    cleaned_text = comment_text.replace('\n', ' ').replace('\r', ' ').replace('"', '').strip()

                    writer.writerow([comment_id, comment_timestamp, comment_permalink, comment_score, comment_controversiality, comment_author, cleaned_text])


# json_to_csv2()

# one-time function to merge three csv files into one, with the same headers as the originals
# used gemini to generate this function.
def merge_csv():
    paths = [
        "src/language_processing/csv/new_piratefolk_comments_2023.csv",
        "src/language_processing/csv/new_piratefolk_comments_2024.csv",
        "src/language_processing/csv/new_piratefolk_comments_2025.csv"
    ]
    df_list = [pd.read_csv(p) for p in paths]
    merged_df = pd.concat(df_list, ignore_index=True)
    output_path = "src/language_processing/csv/new_piratefolk_comments.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Successfully merged {len(paths)} files into {output_path}")


# merge_csv()

# likewise one-time function
def remove_duplicate_ids():
    input_path = "src/language_processing/csv/new_piratefolk_comments.csv"
    output_path = "src/language_processing/csv/new_pf_comments-reduced.csv"
    df = pd.read_csv(input_path)
    df_cleaned = df.drop_duplicates(subset=['id'], keep='first')
    df_cleaned.to_csv(output_path, index=False)
    removed_count = len(df) - len(df_cleaned)
    print(f"Removed {removed_count} duplicate rows. {len(df_cleaned)} rows remaining.")

# remove_duplicate_ids()