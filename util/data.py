import numpy as np
from collections import defaultdict

from dataset_manager import preprocess_dataset


class Dataset(object):
    count = 0

    def __init__(self, filename, limit=None, rebuild=False, use_preprocess=False):
        """
        Wraps dataset and produces batches for the model to consume

        :param filename: path to training data for npz file
        """

        # Se rebuild=False i dataset (train e test) sono identici a prima
        # Ora pero' il dataset di test e' copiato in un nuovo dataset (validation)
        # Se rebuild=True sono alterati i dataset di training e di test.

        if use_preprocess:
            self.train_data, self.test_data = preprocess_dataset()
        else:
            self._data = np.load(filename)
            # print(str(self._data)[:200])

            self.train_data = self._data['train_data'][:, :2]
            self.test_data = self._data['test_data'].tolist()

        # print(self.train_data)
        # print(self.test_data)

        # Shallow copy
        self.validation_data = self.test_data.copy()

        if rebuild:
            # LC > Rebuilding datasets
            # b = len(self.train_data)
            self.negative_items = {}
            for user in self.test_data:
                # self.train_data = np.insert(self.train_data, 0, np.array((user, self.test_data[user][0])), 0)
                self.negative_items[user] = self.test_data[user][1]
                pass
            # print(str(self.negative_items))

            # print('GAP:', (len(self.train_data) - b))
            # assert len(self.train_data) - b == len(self.test_data)
        else:
            # To avoid a weird error
            if not use_preprocess:
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

        if rebuild and not use_preprocess:
            self.test_data = {}
            rows_to_delete = []

            for user in range(limit):
                if user % 1000 == 0:
                    print('Processing user {}'.format(user))

                items = np.array(list(self.user_items[user]), dtype=np.int32)
                items_popularity = self.normalized_popularity[items]
                sorted_items = items[np.argsort(items_popularity)]

                '''
                print('Items for user {}:'.format(user), items)
                print('Items popularity:'.format(user), items_popularity)
                print('Sorted items:'.format(user), sorted_items)
                '''

                try:
                    less_popular_item = sorted_items[0]
                    medium_popular_item = sorted_items[round(len(sorted_items) / 2) - 1]
                    most_popular_item = sorted_items[-1]
                except:
                    print('Items for user {}:'.format(user), items)
                    print('Items popularity:'.format(user), items_popularity)
                    print('Sorted items:'.format(user), sorted_items)
                    pass

                '''
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

    def _sample_negative_item(self, user_id, item=None, index=None, upper_bound=None):
        """
        Uniformly sample a negative item
        """
        if user_id > self.user_count:
            raise ValueError("Trying to sample user id: {} > user count: {}".format(
                user_id, self.user_count))

        positive_items = self.user_items[user_id]
        more_popular_positive_items = set()
        if item is not None:
            # objects more popular than 'item'
            more_popular_positive_items = np.array(list(filter(lambda x: self.normalized_popularity[x] > self.normalized_popularity[item], positive_items)), dtype=np.uint32)

            # ordering more_popular_positive_items by popularity (DESC)
            more_popular_positive_items_popularity = self.normalized_popularity[more_popular_positive_items]
            more_popular_positive_items = more_popular_positive_items[np.argsort(more_popular_positive_items_popularity)]

        if item is None or len(more_popular_positive_items) == 0:
            n = self._sample_item()

            if len(positive_items) >= self.item_count:
                raise ValueError("The User has rated more items than possible %s / %s" % (
                    len(positive_items), self.item_count))
            while n in positive_items or n not in self.item_users:
                n = self._sample_item()
        else:
            # TODO LC > Dealing with popularity
            '''
            select_most_popular_first = False  # True = current experiments (LASCIARE False)
            selected_index = int(index / (upper_bound - 1) * len(more_popular_positive_items))

            if select_most_popular_first:
                selected_index = len(more_popular_positive_items) - 1 - selected_index
            '''
            # print(index)
            if index == 1:  # 0 more, 1 less
                selected_index = len(more_popular_positive_items) - 1
            else:
                selected_index = 0

            n = more_popular_positive_items[selected_index]

            verbose = False
            if verbose:
                print('---------------------------------------')
                print('index:', index)
                print('upper_bound:', upper_bound)
                print('item:', item)
                print('positive_items:', positive_items)
                print('more_popular_positive_items:', more_popular_positive_items)
                print('np.sort(more_popular_positive_items_popularity)):', np.sort(more_popular_positive_items_popularity))
                print('selected_index:', selected_index)
                print('selected item (n):', n)
                print('---------------------------------------')

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
            for i in range(neg_count):
                # TODO > modified by Luciano Caroprese. Now a negative item of a user wrt a positive item is an item not explicitly positive or a positive item with an higher popularity
                if use_popularity:
                    if i % neg_count == (neg_count - 1):
                        # selecting a negative item
                        neg_item_idx = self._sample_negative_item(user_idx)
                    else:
                        # selecting a positive but more popular item (if there is one)
                        neg_item_idx = self._sample_negative_item(user_idx, item_idx, i, neg_count)
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
