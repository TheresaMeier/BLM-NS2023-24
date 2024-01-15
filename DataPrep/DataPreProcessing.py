import json
import datetime
import sys

year = 2020

if year == 2013:
    start_date_1 = datetime.datetime(2013, 7, 13)
    end_date_1 = datetime.datetime(2013, 8, 31)
elif year == 2020:
    start_date_1 = datetime.datetime(2020, 6, 1)
    end_date_1 = datetime.datetime(2020, 7, 31)
else:
    print('Error: Please choose time period')
    sys.exit()
    
# Create a new CSV file for writing
csv_filename = '/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/filtered_tweets_2020.csv'

with open(csv_filename, 'w', encoding='utf-8') as csv_file:
    csv_file.write('tweet_id,tweet_text,hashtags,created_at,retweet_count,reply_count,like_count,quote_count,author_id\n')
    
    # Open the JSONL file for reading
    with open('/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/BLM_raw_total.jsonl', 'r', encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            # Parse each line (JSON object) from the file
            try:
                info = json.loads(line.strip())
                tweet_all_data = info["data"]
                for tweet_data in tweet_all_data:
                    # Extracting the specified fields
                    tweet_id = tweet_data['id']
                    tweet_text = tweet_data['text'].replace('\n', ' ').replace('"', '').replace(',',' ')
                    if tweet_data.get('entities') and tweet_data['entities'].get('hashtags'):
                        hashtags = [tag['tag'] for tag in tweet_data['entities']['hashtags']]
                    else:
                        hashtags = None
                    created_at = datetime.datetime.strptime(tweet_data['created_at'], "%Y-%m-%dT%H:%M:%S.%fZ")
                    retweet_count = tweet_data['public_metrics']['retweet_count']
                    reply_count = tweet_data['public_metrics']['reply_count']
                    like_count = tweet_data['public_metrics']['like_count']
                    quote_count = tweet_data['public_metrics']['quote_count']
                    author_id = tweet_data["author_id"]

                    # Check if the tweet is within the specified date range and has hashtags
                    if start_date_1 <= created_at <= end_date_1 and hashtags:
                        # Write the tweet data to the CSV file
                        csv_file.write(f'{tweet_id},"{tweet_text}","{",".join(hashtags)}",{created_at},{retweet_count},{reply_count},{like_count},{quote_count},{author_id}\n')
            except Exception as e:
                print(f"Error parsing JSON: {e}")
