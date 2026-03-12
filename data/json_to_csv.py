import json
import os
import csv
from pathlib import Path


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
    

with open("piratefolk_comments.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'timestamp', 'score', 'text'])

    directory_path = Path('./data/comments_data')

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
                comment_text = comment.get('body', '')
                comment_score = comment.get('score', 0)

                cleaned_text = comment_text.replace('\n', ' ').replace('\r', ' ').replace('"', '')

                writer.writerow([comment_id, comment_timestamp, comment_score, cleaned_text])

    


    

    
