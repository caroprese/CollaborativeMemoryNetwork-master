import numpy as np
import tensorflow as tf
from tqdm import tqdm

import settings
from settings import Settings, y_custom, sigmoid


def get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                     input_neighborhood_handle,
                     input_neighborhood_lengths_handle,
                     dropout_handle, score_op, max_neighbors, model,

                     return_scores=False):
    """
    test_data = dict([positive, np.array[negatives]])
    """

    out = ''
    scores = []  # n x 100 (punteggi) dove n e' il numero di utenti. Cosa succede se voglio piu' oggetti per ogni utente?
    items = []
    losses = []
    progress = tqdm(test_data.items(), total=len(test_data), leave=False, desc=u'Evaluate || ')

    count = 0

    for user, (pos_list, neg) in progress:
        for pos in pos_list:
            # print('Processing (', user, ' ,', pos, ', ', neg, ')', sep='')

            item_indices = list(neg) + [pos]
            # print('item_indices:',item_indices)

            feed = {
                input_user_handle: [user] * (len(neg) + 1),
                input_item_handle: item_indices,
            }

            # LC > compute loss:
            '''
            user: [u    u    u    ...u   ]
            pos : [p    p    p    ...p   ]
            neg : [n_1  n_2  n_3  ...n_99]

            nei+: [[l+ ][l+ ][l+ ]...[l+ ]] 
            l+  : [l    l    l    ...l    ]

            nei-: [[l-1][l-2][l-3]...[l-99]] 
            l-  : [l_1  l_2  l_3  ...l_99 ]
            '''

            feed_loss = {
                input_user_handle: [user] * len(neg),
                input_item_handle: [pos] * len(neg),
                model.input_items_negative: list(neg)
            }

            if neighborhood is not None:
                # neighborhood: dizionario item: {utenti...}
                neighborhoods, neighborhood_length = np.zeros((len(neg) + 1, max_neighbors), dtype=np.int32), np.ones(len(neg) + 1, dtype=np.int32)
                neighborhoods_negative, neighborhood_length_negative = np.zeros((len(neg), max_neighbors), dtype=np.int32), np.ones(len(neg), dtype=np.int32)
                neighborhoods_positive, neighborhood_length_positive = np.zeros((len(neg), max_neighbors), dtype=np.int32), np.ones(len(neg), dtype=np.int32)

                # vicini degli oggetti [pos, neg_1, ..., neg_99]
                # 100 * max_neighbors
                for _idx, item in enumerate(item_indices):
                    _len_negative = min(len(neighborhood.get(item, [])), max_neighbors)
                    if _len_negative > 0:
                        neighborhoods[_idx, :_len_negative] = neighborhood[item][:_len_negative]
                        neighborhood_length[_idx] = _len_negative
                    else:
                        neighborhoods[_idx, :1] = user
                feed.update({
                    input_neighborhood_handle: neighborhoods,
                    input_neighborhood_lengths_handle: neighborhood_length
                })

                # vicini degli oggetti [neg_1, ..., neg_99]
                # 99 * max_neighbors
                for _idx, item in enumerate(list(neg)):

                    _len_positive = min(len(neighborhood.get(pos, [])), max_neighbors)
                    _len_negative = min(len(neighborhood.get(item, [])), max_neighbors)

                    if _len_positive > 0:
                        neighborhoods_positive[_idx, :_len_positive] = neighborhood[pos][:_len_positive]
                        neighborhood_length_positive[_idx] = _len_positive
                    else:
                        neighborhoods_positive[_idx, :1] = user

                    if _len_negative > 0:
                        neighborhoods_negative[_idx, :_len_negative] = neighborhood[item][:_len_negative]
                        neighborhood_length_negative[_idx] = _len_negative
                    else:
                        neighborhoods_negative[_idx, :1] = user

                feed_loss.update({
                    model.input_neighborhoods: neighborhoods_positive,
                    model.input_neighborhood_lengths: neighborhood_length_positive,

                    model.input_neighborhoods_negative: neighborhoods_negative,
                    model.input_neighborhood_lengths_negative: neighborhood_length_negative,

                    model.input_positive_items_popularity: settings.Settings.normalized_popularity[[pos] * len(neg)],  # Added by LC
                    model.input_negative_items_popularity: settings.Settings.normalized_popularity[list(neg)],  # Added by LC
                })

            score = sess.run(score_op, feed)
            # print('SCORE:', score)
            scores.append(score.ravel())
            items.append(item_indices)
            if return_scores:
                s = ' '.join(["{}:{}".format(n, s) for s, n in zip(score.ravel().tolist(), item_indices)])
                out += "{}\t{}\n".format(user, s)

            loss = sess.run(model.loss, feed_loss)
            losses.append(loss)

            count += 1

    # print('Number of processed tuples:', count)

    if return_scores:
        return scores, items, out, sum(losses) / len(losses)
    return scores, items, sum(losses) / len(losses)


def evaluate_model(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                   input_neighborhood_handle,
                   input_neighborhood_lengths_handle,
                   dropout_handle, score_op, max_neighbors, model,
                   EVAL_AT=[1, 5, 10]):
    scores, items, out, loss = get_model_scores(sess, test_data, neighborhood, input_user_handle, input_item_handle,
                                                input_neighborhood_handle,
                                                input_neighborhood_lengths_handle,
                                                dropout_handle, score_op, max_neighbors, model,
                                                return_scores=True)
    # print('out:', out)
    # print('scores:', scores)

    '''
    
    n = number of users
    scores = n x 100
    items = n x 100
    
    '''

    hrs = []
    custom_hrs = []
    weighted_hrs = []
    ndcgs = []
    hits_list = []
    normalized_hits_list = []
    s = '\n'
    for k in EVAL_AT:
        hr, custom_hr, weighted_hr, ndcg, hits, normalized_hits = get_eval(scores, items, len(scores[0]) - 1, k)

        s += "{:<10} {:<3.4f} " \
             "{:<10} {:<3.4f} " \
             "{:<10} {:<3.4f} " \
             "{:<10} {} " \
             "{:<10} {} " \
             "{:<10} {:.4f}\n". \
            format('HR@%s' % k, hr, 'CUST_HR@%s' % k, custom_hr, 'WEIGH_HR@%s' % k, weighted_hr, 'HITS@%s' % k, hits, 'NORM_HITS@%s' % k, normalized_hits, 'NDCG@%s' % k, ndcg)
        hrs.append(hr)
        custom_hrs.append(custom_hr)
        weighted_hrs.append(weighted_hr)
        ndcgs.append(ndcg)
        hits_list.append(hits)
        normalized_hits_list.append(normalized_hits)
    s += "Avg Loss on Test Set (each loss value is computed on (user, pos, [neg_1, ..., neg_99])): " + str(loss)
    tf.logging.info(s + '\n')

    return hrs, custom_hrs, weighted_hrs, ndcgs, hits_list, normalized_hits_list


def get_eval(scores, items, index, top_n=10):
    """
    if the last element is the correct one, then
    index = len(scores[0])-1
    """
    # print('>>>>>>>>>>>>>>>> scores:', scores)
    # print('>>>>>>>>>>>>>>>> items:', items)
    eps = 1e-15

    ndcg = 0.0
    custom_hr = 0.0
    weighted_hr = 0.0
    hr = 0.0
    hits = np.array([0, 0, 0])

    assert len(scores[0]) > index and index >= 0

    # for score in scores:
    for i in range(len(scores)):
        score = scores[i]
        item_array = items[i]

        # print('index:', i, len(scores))
        # print('score:', score)
        # print('item:', item_array)

        # Get the top n indices
        arg_index = np.argsort(-score)[:top_n]

        if index in arg_index:
            # print('index, arg_index:', index, '-', arg_index)

            current_item = item_array[index]
            current_popularity = Settings.normalized_popularity[current_item]
            current_position = np.where(arg_index == index)[0][0]

            '''
            print('current_item:', current_item)
            print('current_popularity:', current_popularity)
            print('current_position:', current_position)
            print('arg_index:', arg_index)
            '''

            # Get the position
            ndcg += np.log(2.0) / np.log(arg_index.tolist().index(index) + 2.0)

            # Increment
            hr += 1.0

            # Custom HR
            custom_hr += y_custom(current_popularity, current_position, top_n)

            # Custom HR
            weighted_hr += (1 - current_popularity)
            # weighted_hr += sigmoid(1 / (current_popularity + eps))
            if current_popularity <= 0.33:
                hits[0] += 1
            elif current_popularity > 0.33 and current_popularity <= 0.66:
                hits[1] += 1
            else:
                hits[2] += 1

    # print('HITS:', hr)
    # print('len(scores):', len(scores))

    return hr / len(scores), custom_hr / len(scores), weighted_hr / len(scores), ndcg / len(scores), hits, np.around(hits / np.sum(hits), 2)
