#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@author:   Travis A. Ebesu
@created:  2017-03-30
@summary:
'''
import os
import argparse
import random

from tensorflow import set_random_seed

import settings
from settings import set_parameters, get_percentile
from util.helper import get_optimizer_argparse, preprocess_args, create_exp_directory, BaseConfig, get_logging_config
from util.data import Dataset
from util.evaluation import evaluate_model, get_eval, get_model_scores
from util.cmn import CollaborativeMemoryNetwork
import numpy as np
import tensorflow as tf
from logging.config import dictConfig
from tqdm import tqdm
from keras import backend as K, optimizers, metrics

parser = argparse.ArgumentParser(parents=[get_optimizer_argparse()], formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, required=True)
parser.add_argument('--iters', help='Max iters', type=int, default=30)
# parser.add_argument('--iters', help='Max iters', type=int, default=3)
parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=32)
parser.add_argument('-e', '--embedding', help='Embedding Size', type=int, default=50)
parser.add_argument('--dataset', help='path to file', type=str, required=True)
parser.add_argument('--hops', help='Number of hops/layers', type=int, default=2)
parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=4)
# parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=24)
parser.add_argument('--l2', help='l2 Regularization', type=float, default=0.1)
parser.add_argument('-l', '--logdir', help='Set custom name for logdirectory',
                    type=str, default=None)
parser.add_argument('--resume', help='Resume existing from logdir', action="store_true")
parser.add_argument('--pretrain', help='Load pretrained user/item embeddings', type=str,
                    required=True)
parser.set_defaults(optimizer='rmsprop', learning_rate=0.00001, decay=0.9, momentum=0.9)  # se learning_rate=0.0001 le cose vanno meglio
# parser.set_defaults(optimizer='adam', learning_rate=0.001, decay=0.9, momentum=0.9)

FLAGS = parser.parse_args()
preprocess_args(FLAGS)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # FLAGS.gpu

randomness = False
if not randomness:
    # -----------------------------------------------------------------------------
    seed = 42

    np.random.seed(seed)
    random.seed(seed)

    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    # sess
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # tf seed
    tf.set_random_seed(seed)
    # -----------------------------------------------------------------------------

# Create results in here unless we specify a logdir
BASE_DIR = 'result/'
if FLAGS.logdir is not None and not os.path.exists(FLAGS.logdir):
    os.mkdir(FLAGS.logdir)


class Config(BaseConfig):
    logdir = create_exp_directory(BASE_DIR) if FLAGS.logdir is None else FLAGS.logdir
    filename = FLAGS.dataset
    embed_size = FLAGS.embedding
    batch_size = FLAGS.batch_size
    hops = FLAGS.hops
    l2 = FLAGS.l2
    user_count = -1
    item_count = -1
    optimizer = FLAGS.optimizer
    tol = 1e-5
    neg_count = FLAGS.neg
    optimizer_params = FLAGS.optimizer_params
    grad_clip = 5.0
    decay_rate = 0.9
    learning_rate = FLAGS.learning_rate
    pretrain = FLAGS.pretrain
    max_neighbors = -1


config = Config()

if FLAGS.resume:
    config.save_directory = config.logdir
    config.load()

dictConfig(get_logging_config(config.logdir))

# TODO LC > Use the popularity to define a negative item
use_popularity = False
dataset = Dataset(config.filename, limit=None, rebuild=True)  # 55187

config.item_count = dataset.item_count
config.user_count = dataset.user_count
config.save_directory = config.logdir
config.max_neighbors = dataset._max_user_neighbors
tf.logging.info("\n\n%s\n\n" % config)

# LC > settings ---------------------------------------------------------------
set_parameters(
    normalized_popularity=dataset.normalized_popularity,
    loss_alpha=200,
    loss_beta=0.02,
    loss_scale=1,
    loss_percentile=get_percentile(dataset.normalized_popularity, 45),
    metrics_alpha=100,
    metrics_beta=0.03,
    metrics_gamma=5,
    metrics_scale=1 / 15,
    metrics_percentile=0.45,
    loss_type=0
)
# -----------------------------------------------------------------------------

if not FLAGS.resume:
    config.save()

print('CMN Config:', config)
model = CollaborativeMemoryNetwork(config)

sv = tf.train.Supervisor(logdir=config.logdir, save_model_secs=60 * 10,
                         save_summaries_secs=0)

sess = sv.prepare_or_wait_for_session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=True)))

if not FLAGS.resume:
    pretrain = np.load(FLAGS.pretrain)

    sess.graph._unsafe_unfinalize()
    tf.logging.info('Loading Pretrained Embeddings.... from %s' % FLAGS.pretrain)
    sess.run([
        model.user_memory.embeddings.assign(pretrain['user'] * 0.5),
        model.item_memory.embeddings.assign(pretrain['item'] * 0.5)
    ])

# Train Loop
for i in range(FLAGS.iters):
    if sv.should_stop():
        break

    progress = tqdm(enumerate(dataset.get_data(FLAGS.batch_size, True, FLAGS.neg, use_popularity=use_popularity)),
                    dynamic_ncols=True, total=(dataset.train_size * FLAGS.neg) // FLAGS.batch_size)
    loss = []
    for k, example in progress:
        ratings, pos_neighborhoods, pos_neighborhood_length, neg_neighborhoods, neg_neighborhood_length = example
        feed = {
            model.input_users: ratings[:, 0],

            model.input_items: ratings[:, 1],
            model.input_positive_items_popularity: settings.Settings.normalized_popularity[ratings[:, 1]],  # Added by LC

            model.input_items_negative: ratings[:, 2],
            model.input_negative_items_popularity: settings.Settings.normalized_popularity[ratings[:, 2]],  # Added by LC

            model.input_neighborhoods: pos_neighborhoods,
            model.input_neighborhood_lengths: pos_neighborhood_length,
            model.input_neighborhoods_negative: neg_neighborhoods,
            model.input_neighborhood_lengths_negative: neg_neighborhood_length
        }

        verbose = False
        if verbose:
            print('\nBATCH ----------------------------')
            print('input_users:\n', ratings[:, 0])
            print('input_items:\n', ratings[:, 1])
            # print('input_positive_items_popularity:\n', settings.Settings.normalized_popularity[ratings[:, 1]])
            print('input_items_negative:\n', ratings[:, 2])
            # print('input_negative_items_popularity:\n', settings.Settings.normalized_popularity[ratings[:, 2]])
            # print('input_negative_items_popularity/input_positive_items_popularity:\n',
            #      settings.Settings.normalized_popularity[ratings[:, 2]] / settings.Settings.normalized_popularity[ratings[:, 1]])
            print('pos_neighborhoods:\n', pos_neighborhoods)
            print('pos_neighborhood_length:\n', pos_neighborhood_length)
            print('neg_neighborhoods:\n', neg_neighborhoods)
            print('neg_neighborhood_length:\n', neg_neighborhood_length)

        batch_loss, _ = sess.run([model.loss, model.train], feed)
        loss.append(batch_loss)
        progress.set_description(u"[{}] Loss (type={}): {:,.4f} » » » » ".format(i, settings.Settings.loss_type, batch_loss))

    tf.logging.info("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))
    evaluate_model(sess,
                   dataset.test_data,
                   dataset.item_users_list,
                   model.input_users,
                   model.input_items,
                   model.input_neighborhoods,
                   model.input_neighborhood_lengths,
                   model.dropout,
                   model.score,
                   config.max_neighbors,
                   model)

EVAL_AT = range(1, 11)
hrs, custom_hrs, weighted_hrs, ndcgs, hits_list, normalized_hits_list = [], [], [], [], [], []
s = ""
scores, items, out, loss = get_model_scores(sess,
                                            dataset.test_data,
                                            dataset.item_users_list,
                                            model.input_users,
                                            model.input_items,
                                            model.input_neighborhoods,
                                            model.input_neighborhood_lengths,
                                            model.dropout,
                                            model.score,
                                            config.max_neighbors,
                                            model,
                                            True)

for k in EVAL_AT:
    hr, custom_hr, weighted_hr, ndcg, hits, normalized_hits = get_eval(scores, items, len(scores[0]) - 1, k)
    hrs.append(hr)
    custom_hrs.append(custom_hr)
    weighted_hrs.append(weighted_hr)
    hits_list.append(hits)
    normalized_hits_list.append(normalized_hits)
    ndcgs.append(ndcg)
    s += "{:<10} {:<3.4f} " \
         "{:<10} {:<3.4f} " \
         "{:<10} {:<3.4f} " \
         "{:<10} {} " \
         "{:<10} {} " \
         "{:<10} {:.4f}\n".format('HR@%s' % k, hr,
                                  'CUSTOM_HR@%s' % k, custom_hr,
                                  'WEIGHTED_HR@%s' % k, weighted_hr,
                                  'HITS@%s' % k, str(hits),
                                  'NORM_HITS@%s' % k, str(normalized_hits),
                                  'NDCG@%s' % k, ndcg)
s += "Avg Loss on Test Set (each loss value is computed on (user, pos, [neg_1, ..., neg_99])): " + str(loss)
tf.logging.info(s)

with open("{}/final_results".format(config.logdir), 'w') as fout:
    header = ','.join([str(k) for k in EVAL_AT])
    fout.write("{},{}\n".format('metric', header))
    ndcg = ','.join([str(x) for x in ndcgs])
    custom_hr = ','.join([str(x) for x in custom_hrs])
    hr = ','.join([str(x) for x in hrs])
    fout.write("ndcg,{}\n".format(ndcg))
    fout.write("custom_hr,{}\n".format(custom_hr))
    fout.write("hr,{}".format(hr))

tf.logging.info("Saving model...")
# Save before exiting
sv.saver.save(sess, sv.save_path, global_step=tf.contrib.framework.get_global_step())
sv.request_stop()
