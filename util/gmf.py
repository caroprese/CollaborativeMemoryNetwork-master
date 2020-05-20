import sonnet as snt
import tensorflow as tf

import settings
from util.helper import GraphKeys, add_to_collection
from util.layers import DenseLayer, LossLayer, OptimizerLayer, ModelBase


class PairwiseGMF(ModelBase):

    def __init__(self, config):
        """

        :param config:
        """
        # super(PairwiseGMF, self).__init__(config)
        self.config = config
        self._activation_fn = tf.nn.relu
        self._embedding_initializers = {
            'embeddings': tf.truncated_normal_initializer(stddev=0.01),
        }

        self._embedding_regularizers = {}

        self._initializers = {
            "w": tf.contrib.layers.xavier_initializer(),
        }

        self._regularizers = {
            'w': tf.contrib.layers.l2_regularizer(config.l2)
        }

        self._construct_placeholders()
        self._construct_weights()
        self._construct()
        tf.summary.scalar('Model/Loss', tf.get_collection(GraphKeys.LOSSES)[0])
        self.summary = tf.summary.merge_all()

    def _construct(self):
        """
        Construct the model; main part of it goes here
        """

        self.v = DenseLayer(1, False, tf.nn.relu, initializers=self._initializers,
                                            regularizers=self._regularizers, name='OutputVector')
        self.score = tf.squeeze(self.v(self._cur_user * self._cur_item))
        self.negative_output = tf.squeeze(self.v(self._cur_user * self._cur_item_negative))
        tf.add_to_collection(GraphKeys.PREDICTION, self.score)

        parameter_k = None
        if settings.Settings.loss_type == 2:
            self.k = tf.Variable(settings.Settings.k, trainable=True, dtype=tf.float32, name='k')
            parameter_k = self.k

        self.loss = LossLayer()(self.score,
                                self.negative_output,
                                self.input_positive_items_popularity,
                                self.input_negative_items_popularity,
                                parameter_k)

        self._optimizer = OptimizerLayer(self.config.optimizer, clip=5.0,
                                         params={})
        self.train = self._optimizer(self.loss)

    def _construct_weights(self):
        """
        Constructs the user/item memories and user/item external memory/outputs

        Also add the embedding lookups
        """
        self.user_memory = snt.Embed(self.config.user_count, self.config.embed_size,
                                     initializers=self._embedding_initializers,
                                     regularizers=self._embedding_regularizers,
                                     name='MemoryEmbed')

        self.item_memory = snt.Embed(self.config.item_count,
                                     self.config.embed_size,
                                     initializers=self._embedding_initializers,
                                     regularizers=self._embedding_regularizers,
                                     name="ItemMemory")

        # [batch, embedding size]
        self._cur_user = self.user_memory(self.input_users)

        # Item memories a query
        self._cur_item = self.item_memory(self.input_items)
        self._cur_item_negative = self.item_memory(self.input_items_negative)

    def _construct_placeholders(self):
        self.input_users = tf.placeholder(tf.int32, [None], 'UserID')
        self.input_items = tf.placeholder(tf.int32, [None], 'ItemID')

        # LC > popularity of positive items
        self.input_positive_items_popularity = tf.placeholder(tf.float32, [None], 'PositiveItemsPopularity')

        self.input_items_negative = tf.placeholder(tf.int32, [None], 'NegativeItemID')

        # LC > popularity of negative items
        self.input_negative_items_popularity = tf.placeholder(tf.float32, [None], 'NegativeItemsPopularity')

        # Add our placeholders
        add_to_collection(GraphKeys.PLACEHOLDER, [self.input_users,
                                                  self.input_items,
                                                  self.input_positive_items_popularity,
                                                  self.input_items_negative,
                                                  self.input_negative_items_popularity])