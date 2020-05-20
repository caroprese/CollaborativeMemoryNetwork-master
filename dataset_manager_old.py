from collections import defaultdict
import random
import pandas as pd

seed = 42
random.seed(seed)


def preprocess_dataset_old(filename='./data/ml-1m/ratings.dat'):
    return None, None


def preprocess_dataset_old(filename='./data/ml-1m/ratings.dat',
                           names=['user', 'item', 'rating', 'ts'],
                           item_col_name='item',
                           user_col_name='user',
                           sep="::",
                           min_users_per_item=5,
                           min_items_per_user=10):
    df = pd.read_csv(filename, engine='python', names=names, sep=sep)
    df['user'] = df['user'] - 1
    df['item'] = df['item'] - 1
    df_test = df.groupby(user_col_name).filter(lambda group: len(group[group['rating'] > 3.5]) < min_items_per_user)
    print("..........................")
    print(df_test[:100])
    print("..........................")

    df = df.groupby(item_col_name).filter(lambda group: len(group[group['rating'] > 3.5]) >= min_users_per_item)
    df = df.groupby(user_col_name).filter(lambda group: len(group[group['rating'] > 3.5]) >= min_items_per_user)

    data = df[['user', 'item', 'rating']].to_numpy()
    # print(data)

    user_items = defaultdict(set)
    item_users = defaultdict(set)
    user_positive_items = defaultdict(set)
    positive_item_users = defaultdict(set)

    for user, item, rating in data:
        user_items[user].add(item)
        item_users[item].add(user)
        if rating > 3.5:
            user_positive_items[user].add(item)
            positive_item_users[item].add(user)

    print(">>>>>>>>>>>>>>>>>>>>>>>>")
    print(user_items[3597])
    print(user_positive_items[3597])

    n_users, n_items, _ = data.max(axis=0) + 1

    print(n_users)
    print(n_items)
    items = {item for item in range(n_items)}

    test_data = {}
    for user in range(n_users):
        positive_items = list(user_positive_items[user])
        negative_items = random.sample(items.difference(user_items[user]), 99)
        '''
        print('User {}'.format(user))
        print('pos:', user_items[user])
        print('neg:', negative_items)
        print('intersection:', user_items[user].intersection(negative_items))
        '''
        assert len(user_items[user].intersection(negative_items)) == 0

        test_data[user] = (positive_items, negative_items)

    train_data = df[df['rating'] > 3.5][['user', 'item']].to_numpy()

    return train_data, test_data
