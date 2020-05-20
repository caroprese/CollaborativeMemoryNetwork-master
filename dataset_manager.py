from collections import defaultdict
import random
import pandas as pd
import json
import numpy as np

seed = 42
random.seed(seed)


def preprocess_dataset(filename='./data/dataset_movielens_pop_05_25.json'):
    with open(filename) as json_file:
        data = json.load(json_file)

    n_users = data['n_users']
    n_items = data['n_items']

    print('n_users:', n_users)
    print('n_items:', n_items)

    ds = data['dataset']
    dataset_test = data['dataset_test']

    # print('ds.len:', len(ds))
    # print(str(dataset_test)[:1000])

    # print(type(ds))
    # print(ds)
    train_data = []
    test_data = {}
    for user in range(len(ds)):
        items = ds[user]
        # print(items)
        positive_test_items = dataset_test[user][0]
        negative_test_items = dataset_test[user][1]
        # print('Test Items:', test_items)
        for item in items:
            if item not in positive_test_items:
                train_data.append([user, item])
        test_data[user] = (positive_test_items, negative_test_items)

    # print(str(dataset)[:100])
    train_data = np.array(train_data)

    print(train_data)

    return train_data, test_data


preprocess_dataset()
