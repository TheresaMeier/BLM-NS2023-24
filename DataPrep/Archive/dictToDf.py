import json
import pickle
import pandas as pd

pickle_file_path ='/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/BLM_raw_2013.pkl'

with open(pickle_file_path, 'rb') as file:
    # Load the data from the pickle file
    data_raw = pickle.load(file)


df = pd.DataFrame()


if data_raw:
    # Iterate through each dictionary in the list
    for entry in range(len(data_raw)):
        # Convert the dictionary to a DataFrame and append it
        author_id = data_raw[entry]['data'][0]['author_id']
        text = data_raw[entry]['data'][0]['text']
        hashtags = []
        for tag in data_raw[entry]['data'][0]['entities']['hashtags']:
            hashtags.append(data_raw[entry]['data'][0]['entities']['hashtags'][tag]['tag'])

        retweets = data_raw[entry]['data'][0]['public_metrics']['retweet_count']
        replies = data_raw[entry]['data'][0]['public_metrics']['reply_count']
        likes = data_raw[entry]['data'][0]['public_metrics']['like_count']
        quotes = data_raw[entry]['data'][0]['public_metrics']['quote_count']
        entry_df = pd.DataFrame({'author_id': [author_id], 'text':[text], 'retweet_count': [retweets],'reply_count': [replies],'like_count': [likes], 'quote_count': [quotes], 'hashtags':[hashtags]})
        df = pd.concat([df, entry_df], ignore_index=True)

    # Display the resulting DataFrame
    print(df)