import numpy as np
from collections import defaultdict


class Dataset(object):
    count = 0

    def __init__(self, filename, limit=None, rebuild=False):
        """
        Wraps dataset and produces batches for the model to consume

        :param filename: path to training data for npz file
        """
        self._data = np.load(filename)

        self.train_data = self._data['train_data'][:, :2]
        self.test_data = self._data['test_data'].tolist()

        if rebuild:
            # LC > Rebuilding datasets
            b = len(self.train_data)
            self.negative_items = {}
            for user in self.test_data:
                self.train_data = np.insert(self.train_data, 0, np.array((user, self.test_data[user][0])), 0)
                self.negative_items[user] = self.test_data[user][1]
            # print(str(self.negative_items))

            # print('GAP:', (len(self.train_data) - b))
            assert len(self.train_data) - b == len(self.test_data)
        else:
            for user in self.test_data:
                self.test_data[user] = ([self.test_data[user][0]], self.test_data[user][1])

        self._n_users, self._n_items = self.train_data.max(axis=0) + 1

        # LC > (Normalized) Popularity ----------------------------------------
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in self.train_data:
            self.user_items[u].add(i)
            self.item_users[i].add(u)

        self.popularity = np.zeros(self._n_items)
        for i in self.item_users:
            self.popularity[i] = len(self.item_users[i])
            # print('LC > popularity of item n.', i, ':', len(self.item_users[i]))

        min_value = np.min(self.popularity)
        max_value = np.max(self.popularity)
        gap = max_value - min_value

        self.normalized_popularity = (self.popularity - min_value) / gap
        # ---------------------------------------------------------------------

        # LC > Implementing limit ---------------------------------------------
        if limit is None:
            limit = self._n_users
        self.train_data = self.train_data[self.train_data[:, 0] < limit]
        self.test_data = {key: self.test_data[key] for key in range(limit)}

        # print(self.train_data)
        # print(self.test_data)

        # print(self.train_data[0:10,:])
        # print('TEST SET: ------------------------------------------------------')
        # print(type(self.test_data))
        # print(len(self.test_data.keys()))
        # print(list(self.test_data.keys())[:100])
        # ---------------------------------------------------------------------

        if rebuild:
            self.test_data = {}
            rows_to_delete = []

            for user in range(limit):
                if user % 1000 == 0:
                    print('Processing user {}'.format(user))

                items = np.array(list(self.user_items[user]))
                items_popularity = self.normalized_popularity[items]
                sorted_items = items[np.argsort(items_popularity)]

                less_popular_item = sorted_items[0]
                medium_popular_item = sorted_items[round(len(sorted_items) / 2) - 1]
                most_popular_item = sorted_items[-1]

                '''
                print('Items for user {}:'.format(user), items)
                print('Items popularity:'.format(user), items_popularity)
                print('Sorted items:'.format(user), sorted_items)
                print('Less popular object:', less_popular_item)
                print('Medium popular object:', medium_popular_item)
                print('Most popular object:', most_popular_item)
                '''

                rows_to_delete.append([user, less_popular_item])
                rows_to_delete.append([user, medium_popular_item])
                rows_to_delete.append([user, most_popular_item])

                '''
                # Updating training set
                for item in (less_popular_item, medium_popular_item, most_popular_item):
                    for i in range(self.train_data.shape[0]):
                        if np.array_equal(self.train_data[i], np.array([user, item])):
                            print('deleting {} from training set'.format(self.train_data[i]))
                            self.train_data = np.delete(self.train_data, i, 0)
                            break
                '''
                self.test_data[user] = ([less_popular_item, medium_popular_item, most_popular_item], self.negative_items[user])

            rows_to_delete = np.array(rows_to_delete, dtype=np.uint32)

            a1_rows = self.train_data.view([('', self.train_data.dtype)] * self.train_data.shape[0 if self.train_data.flags['F_CONTIGUOUS'] else -1])
            a2_rows = rows_to_delete.view([('', rows_to_delete.dtype)] * rows_to_delete.shape[0 if rows_to_delete.flags['F_CONTIGUOUS'] else -1])  # 1

            print('before:\n', self.train_data)
            self.train_data = np.setdiff1d(a1_rows, a2_rows).view(self.train_data.dtype).reshape(-1, self.train_data.shape[1])
            print('after:\n', self.train_data)

            # print('after:', self.train_data[self.train_data[:, 0] == 0])

        # print('TEST DATA:', self.test_data)
        self._train_index = np.arange(len(self.train_data), dtype=np.uint)

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)
        for u, i in self.train_data:
            self.user_items[u].add(i)
            self.item_users[i].add(u)

        # Get a list version so we do not need to perform type casting
        self.item_users_list = {k: list(v) for k, v in self.item_users.items()}
        self._max_user_neighbors = max([len(x) for x in self.item_users.values()])
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

    @property
    def train_size(self):
        """
        :return: number of examples in training set
        :rtype: int
        """
        return len(self.train_data)

    @property
    def user_count(self):
        """
        Number of users in dataset
        """
        return self._n_users

    @property
    def item_count(self):
        """
        Number of items in dataset
        """
        return self._n_items

    def _sample_item(self):
        """
        Draw an item uniformly
        """
        return np.random.randint(0, self.item_count)

    def _sample_negative_item(self, user_id, item=None):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        n = self._sample_item()

        positive_items = self.user_items[user_id]

        # TODO LC > Dealing with popularity
        if item is not None:
            # print('ITEM:', item)
            # print('BEFORE:', positive_items)
            positive_items = set(filter(lambda x: self.normalized_popularity[x] <= self.normalized_popularity[item], positive_items))
            # print('AFTER:', positive_items)

        if len(positive_items) >= self.item_count:
            raise ValueError("The User has rated more items than possible %s / %s" % (
                len(positive_items), self.item_count))
        while n in positive_items or n not in self.item_users:
            n = self._sample_item()
        return n

    def _generate_data(self, neg_count):
        idx = 0
        self._examples = np.zeros((self.train_size * neg_count, 3),
                                  dtype=np.uint32)
        self._examples[:, :] = 0
        for user_idx, item_idx in self.train_data:
            for _ in range(neg_count):
                neg_item_idx = self._sample_negative_item(user_idx)
                self._examples[idx, :] = [user_idx, item_idx, neg_item_idx]
                idx += 1

    def get_data(self, batch_size: int, neighborhood: bool, neg_count: int, use_popularity=False):
        """
        Batch data together as (user, item, negative item), pos_neighborhood,
        length of neighborhood, negative_neighborhood, length of negative neighborhood

        if neighborhood is False returns only user, item, negative_item so we
        can reuse this for non-neighborhood-based methods.

        :param batch_size: size of the batch
        :param neighborhood: return the neighborhood information or not
        :param neg_count: number of negative samples to uniformly draw per a pos
                          example
        :return: generator
        """
        # Allocate inputs
        batch = np.zeros((batch_size, 3), dtype=np.uint32)
        pos_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        pos_length = np.zeros(batch_size, dtype=np.int32)
        neg_neighbor = np.zeros((batch_size, self._max_user_neighbors), dtype=np.int32)
        neg_length = np.zeros(batch_size, dtype=np.int32)

        # Shuffle index
        np.random.shuffle(self._train_index)

        idx = 0
        for user_idx, item_idx in self.train_data[self._train_index]:
            # TODO: set positive values outside of for loop
            for _ in range(neg_count):
                # TODO > modified by Luciano Caroprese. Now a negative item of a user wrt a positive item is an item not explicitly positive or a positive item with an higher popularity
                if use_popularity:
                    neg_item_idx = self._sample_negative_item(user_idx, item_idx)
                else:
                    neg_item_idx = self._sample_negative_item(user_idx)
                batch[idx, :] = [user_idx, item_idx, neg_item_idx]

                # Get neighborhood information
                if neighborhood:
                    if len(self.item_users.get(item_idx, [])) > 0:
                        pos_length[idx] = len(self.item_users[item_idx])
                        pos_neighbor[idx, :pos_length[idx]] = self.item_users_list[item_idx]
                    else:
                        # Length defaults to 1
                        pos_length[idx] = 1
                        pos_neighbor[idx, 0] = item_idx

                    if len(self.item_users.get(neg_item_idx, [])) > 0:
                        neg_length[idx] = len(self.item_users[neg_item_idx])
                        neg_neighbor[idx, :neg_length[idx]] = self.item_users_list[neg_item_idx]
                    else:
                        # Length defaults to 1
                        neg_length[idx] = 1
                        neg_neighbor[idx, 0] = neg_item_idx

                idx += 1
                # Yield batch if we filled queue
                if idx == batch_size:
                    if neighborhood:
                        max_length = max(neg_length.max(), pos_length.max())
                        yield batch, pos_neighbor[:, :max_length], pos_length, \
                              neg_neighbor[:, :max_length], neg_length
                        pos_length[:] = 1
                        neg_length[:] = 1
                    else:
                        yield batch
                    # Reset
                    idx = 0

        # Provide remainder
        if idx > 0:
            if neighborhood:
                max_length = max(neg_length[:idx].max(), pos_length[:idx].max())
                yield batch[:idx], pos_neighbor[:idx, :max_length], pos_length[:idx], \
                      neg_neighbor[:idx, :max_length], neg_length[:idx]
            else:
                yield batch[:idx]
