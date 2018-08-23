import tensorflow as tf
from module import *
from data import load_vocab
from hyperparams import Hyperparams as hp


class Graph():
    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Input
            self.x = tf.placeholder(tf.int32, shape=(None, hp.max_len,))
            self.y = tf.placeholder(tf.int32, shape=(None, hp.max_len,))

            batch_size = tf.shape(tf.reduce_sum(tf.to_int32(tf.not_equal(self.x, hp.vocab_size + 1)), 1))[0]
            # Vocab
            word_w2i, _ = load_vocab()
            self.word_embeddings = embed(self.x, len(word_w2i), hp.word_embed_size)
            h, res_input = self.word_embeddings, self.word_embeddings

            for i in range(hp.cnn_layers):
                # residual connection
                if i % hp.block_size == 0:
                    h += res_input
                    res_input = h
                # zero-padding for first layer
                if i == 0:
                    h = down_shift(self.word_embeddings)
                    h = right_shift(h)
                # zero padding for rest layers
                else:
                    h = down_shift(h)
                    h = right_shift(h)
                fanin_depth = h.get_shape()[-1]
                filter_size = hp.filter_size if i < hp.cnn_layers - 1 else 1
                shape = (hp.filter_h, hp.filter_w, fanin_depth, filter_size)
                with tf.variable_scope("layer_%d" % i):
                    conv_linear = convolution(h, shape, "linear")
                    conv_gated = convolution(h, shape, "gated")
                    h = conv_linear * tf.sigmoid(conv_gated)

            self.h = tf.reshape(h, (-1, hp.word_embed_size))

            self.softmax_w = tf.get_variable("softmax_w", [hp.vocab_size, hp.word_embed_size], tf.float32,
                                             tf.random_normal_initializer(0.0, 0.1))
            self.softmax_b = tf.get_variable("softmax_b", [hp.vocab_size], tf.float32, tf.constant_initializer(1.0))

            # loss
            # Preferance: NCE Loss, hierarchial softmax, adaptive softmax
            self.loss = tf.reduce_mean(tf.nn.nce_loss(self.softmax_w,
                                                      self.softmax_b,
                                                      inputs=self.h,
                                                      labels=tf.reshape(self.y, (batch_size * hp.max_len, 1)),
                                                      num_sampled=hp.num_nce_sampled,
                                                      num_classes=hp.vocab_size))
            self.perplexity = tf.exp(self.loss)

            # optimizing
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            gradients = tf.train.MomentumOptimizer(hp.lr, hp.momentum).compute_gradients(self.loss)
            clipped_gradients = [(tf.clip_by_value(_[0], -hp.grad_clip, hp.grad_clip), _[1]) for _ in gradients]
            self.optimizer = tf.train.MomentumOptimizer(hp.lr, hp.momentum).apply_gradients(clipped_gradients,
                                                                                            self.global_step)
