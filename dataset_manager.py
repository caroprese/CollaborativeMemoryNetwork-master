from collections import defaultdict
import random
import pandas as pd
import json
import numpy as np
import pickle


def preprocess_dataset(path='./data/preprocess/ml-1m/pg/'):
    with open(path + 'train.pickle', 'rb') as f:
        train_data_source = pickle.load(f)
    # print(data)

    users = train_data_source['users']
    popularity = train_data_source['popularity']
    print(popularity)
    exit(0)
    thresholds = train_data_source['thresholds']
    train_data_dict = train_data_source['prefs']
    items = max(popularity)

    # print(users)
    # print(items)
    # print(thresholds)
    # print(prefs)
    train_data = []
    for user in train_data_dict:
        # print('pos,neg:', train_data_dict[user])

        # print('pos', train_data_dict[user][0])
        for item in train_data_dict[user][0]:
            # print('[{},{}]'.format(user, item))
            train_data.append([user, item, train_data_dict[user][1]])

    # print(popularity)

    with open(path + 'test_te.pickle', 'rb') as f:
        test_data_source = pickle.load(f)

    test_data = test_data_source['prefs']

    print(train_data)
    #print(test_data)

    return train_data, test_data


preprocess_dataset()
