'''
@author:   Travis A. Ebesu
@created:  2017-03-30
@summary:
'''

import argparse
import os
import numpy as np
import tensorflow as tf
import settings

from settings import set_parameters, get_percentile
from util.gmf import PairwiseGMF
from util.helper import BaseConfig
from util.data import Dataset
from tqdm import tqdm

CITEULIKE = 'citeulike-a'
EPINIONS = 'epinions'
MOVIELENS_1M = 'ml-1m'
MOVIELENS_20M = 'ml-20m'
NETFLIX_SAMPLE = 'netflix_sample'
PINTEREST = 'pinterest'
NETFLIX = 'netflix'

datasets_to_pretrain = [EPINIONS, CITEULIKE, MOVIELENS_20M, NETFLIX_SAMPLE, PINTEREST, NETFLIX]
types = [True, False]

# Base Parameters -------------------------------
for d in datasets_to_pretrain:
    for baseline in types:
        print('PRETRAINING ----------------------------------------------------')
        print(d)
        if baseline:
            print('Baseline')
        else:
            print('Our Approach')
        print('----------------------------------------------------------------')
        gpu = '2'
        epochs = 15
        limit = None
        users_per_batch = 5

        neg_items = 2  # baseline=4 (our setting=2)
        use_popularity = True  # if True, the training set contains samples of the form [user, pos, pos'], where pos' is a positive item for u more popular than pos

        loss_type = 2  # baseline=0
        learning_rate = 0.00001  # baseline=0.001 (our setting=0.00001)

        load_pretrained_embeddings = False
        # -----------------------------------------------

        # Derived Parameters ----------------------------
        resume = False
        logdir = None

        if d == PINTEREST:
            batch_size = 256
        else:
            batch_size = 128

        k = 300  # a parameter for the new loss
        k_trainable = False

        loss_alpha = 200
        loss_beta = 0.02
        loss_scale = 1
        metrics_alpha = 100
        metrics_beta = 0.03
        metrics_gamma = 5
        metrics_scale = 1 / 15
        metrics_percentile = 0.45

        # BASELINE --------------------------------------
        if baseline:
            use_popularity = False  # [Evaluation phase] If use_popularity==True, a negative item N wrt a positive item P, can be a positive item with a lower popularity than P
            loss_type = 0
            neg_items = 4
            learning_rate = 0.001
        # -----------------------------------------------
        # -----------------------------------------------

        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('-g', '--gpu', help='set gpu device number 0-3', type=str, required=True)
        parser.add_argument('--iters', help='Max iters', type=int, default=epochs)
        parser.add_argument('-b', '--batch_size', help='Batch Size', type=int, default=batch_size)
        parser.add_argument('-e', '--embedding', help='Embedding Size', type=int, default=50)
        parser.add_argument('--dataset', help='path to npz file', type=str, required=True)
        parser.add_argument('-n', '--neg', help='Negative Samples Count', type=int, default=4)
        parser.add_argument('--l2', help='l2 Regularization', type=float, default=0.001)
        parser.add_argument('-o', '--output', help='save filename for trained embeddings', type=str, required=True)

        FLAGS = parser.parse_args()

        # Base Parameters -------------------------------
        FLAGS.gpu = gpu
        FLAGS.dataset = 'data/preprocess/' + d + '/'
        FLAGS.output = './pretrain/' + d + '_loss_' + str(loss_type) + '.npz'

        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu


        class Config(BaseConfig):
            filename = FLAGS.dataset
            embed_size = FLAGS.embedding
            batch_size = FLAGS.batch_size
            l2 = FLAGS.l2
            user_count = -1
            item_count = -1
            optimizer = 'adam'
            neg_count = FLAGS.neg
            learning_rate = 0.001


        config = Config()

        dataset = Dataset(config.filename, limit=limit)
        set_parameters(
            normalized_popularity=dataset.normalized_popularity,
            loss_alpha=loss_alpha,
            loss_beta=loss_beta,
            loss_scale=loss_scale,
            loss_percentile=get_percentile(dataset.normalized_popularity, 45),
            metrics_alpha=metrics_alpha,
            metrics_beta=metrics_beta,
            metrics_gamma=metrics_gamma,
            metrics_scale=metrics_scale,
            metrics_percentile=metrics_percentile,
            loss_type=loss_type,
            k=k,
            k_trainable=k_trainable,
            low_popularity_threshold=dataset.thresholds[0],
            high_popularity_threshold=dataset.thresholds[1]
        )

        # -----------------------------------------------------------------------------

        config.item_count = dataset.item_count
        config.user_count = dataset.user_count
        tf.logging.info("\n\n%s\n\n" % config)

        model = PairwiseGMF(config)
        sv = tf.train.Supervisor(logdir=None,
                                 save_model_secs=0,
                                 save_summaries_secs=0)
        sess = sv.prepare_or_wait_for_session(
            config=tf.ConfigProto(gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.1,
                allow_growth=True)))

        for i in range(FLAGS.iters):
            if sv.should_stop():
                break
            progress = tqdm(enumerate(dataset.get_data(FLAGS.batch_size, False, FLAGS.neg)),
                            dynamic_ncols=True, total=(dataset.train_size * FLAGS.neg) // FLAGS.batch_size)
            loss = []
            for k, example in progress:
                feed = {
                    model.input_users: example[:, 0],
                    model.input_items: example[:, 1],
                    model.input_items_negative: example[:, 2],
                    model.input_positive_items_popularity: settings.Settings.normalized_popularity[example[:, 1]],  # Added by LC
                    model.input_negative_items_popularity: settings.Settings.normalized_popularity[example[:, 2]]  # Added by LC
                }
                if settings.Settings.loss_type == 2:
                    batch_loss, _, parameter_k = sess.run([model.loss, model.train, model.k], feed)
                else:
                    parameter_k = 0
                    batch_loss, _ = sess.run([model.loss, model.train], feed)

                loss.append(batch_loss)
                progress.set_description(u"[{}] Loss (type={}, k={}): {:,.4f} » » » » ".format(i, settings.Settings.loss_type, parameter_k, batch_loss))

            print("Epoch {}: Avg Loss/Batch {:<20,.6f}".format(i, np.mean(loss)))

        user_embed, item_embed, v = sess.run([model.user_memory.embeddings, model.item_memory.embeddings, model.v.w])
        np.savez(FLAGS.output, user=user_embed, item=item_embed, v=v)
        print('Saving to: %s' % FLAGS.output)
        sv.request_stop()
print('Done!')
