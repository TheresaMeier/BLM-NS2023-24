import json
# import numpy as np
# import pandas as pd
import datetime
import pickle
# import time # if unix time is needed

if __name__=="__main__":
    data_path_input = "/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/BLM_raw_total.jsonl"
    data_path_output = '/home/theresa/Schreibtisch/Theresa/STUDIUM/Master Statistics and Data Science/Padova/Network Science/Project/Data/BLM_raw_2022_1000.pkl'
    min_retweeds = 1000

    start_date_1 = datetime.datetime(2013, 7, 13)
    end_date_1 = datetime.datetime(2013, 8, 31)

    start_date_2 = datetime.datetime(2020, 6, 1)
    end_date_2 = datetime.datetime(2020, 7, 31)
    
    data_collection_list = []
    with open(data_path_input, 'r') as file:
        for line in file:
            line_out = json.loads(line) 
            for post in range(len(line_out)):
                date_list = line_out['data'][post]['created_at'].split('T')[0].split('-')
                date_list = [int(i) for i in date_list]
                date = datetime.datetime(*date_list)
                retweets = line_out['data'][post]['public_metrics']['retweet_count']
                if date > start_date_1 and date < end_date_1:
                   data_collection_list.append(line_out)
                #if date > start_date_2 and date < end_date_2 and retweets >= min_retweeds:
                 #   data_collection_list.append(line_out['data'])
                else:
                    continue

    with open(data_path_output, 'wb') as f:
        pickle.dump(data_collection_list, f)
