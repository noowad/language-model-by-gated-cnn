import tensorflow as tf
from hyperparams import Hyperparams as hp


def embed(x, vocab_size, num_units, trainable=True):
    # word embedding
    lookup_table = tf.get_variable('lookup_table',
                                   dtype=tf.float32,
                                   shape=[vocab_size, num_units],
                                   trainable=trainable,
                                   initializer=tf.random_uniform_initializer(-1.0, 1.0))
    embeds = tf.nn.embedding_lookup(lookup_table, x)
    batch_size = tf.shape(tf.reduce_sum(tf.to_int32(tf.not_equal(x, hp.vocab_size + 1)), 1))[0]
    embeds = tf.reshape(embeds, (batch_size, hp.max_len, hp.word_embed_size, 1))
    return embeds


def down_shift(x):
    x_shape = x.get_shape().as_list()
    # for zero-padding
    batch_size = tf.shape(tf.reduce_sum(tf.to_int32(tf.not_equal(x, hp.vocab_size + 1)), 1))[0]
    return tf.concat((x[:, :, :, :], tf.zeros([batch_size, hp.max_len, hp.filter_h - 1, x_shape[3]])), 2)


def right_shift(x):
    x_shape = x.get_shape().as_list()
    # for zero-padding
    batch_size = tf.shape(tf.reduce_sum(tf.to_int32(tf.not_equal(x, hp.vocab_size + 1)), 1))[0]
    return tf.concat((tf.zeros([batch_size, hp.filter_h - 1, hp.word_embed_size + hp.filter_h - 1, x_shape[3]]),
                      x[:, :, :, :]), 1)


def convolution(fan_in, shape, name):
    W = tf.get_variable("%s_W" % name, shape, tf.float32, tf.random_normal_initializer(0.0, 0.1))
    b = tf.get_variable("%s_b" % name, shape[-1], tf.float32, tf.constant_initializer(1.0))
    return tf.add(tf.nn.conv2d(fan_in, W, strides=[1, 1, 1, 1], padding='VALID'), b)
