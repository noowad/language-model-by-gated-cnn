import tensorflow as tf
import os
from tqdm import tqdm
import random
from data import load_sentences, create_data, load_vocab
from hyperparams import Hyperparams as hp
from graph import Graph
import argparse
import numpy as np

if __name__ == '__main__':
    if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)

    # data load
    sents = load_sentences(hp.data_path + '/1-billion-word.train', False)
    valid_sents = load_sentences(hp.data_path + '/1-billion-word.dev', False)
    _, i2w = load_vocab()

    print('Creating datas...')
    X, Y, _, _ = create_data(sents)
    valid_X, valid_Y, _, _ = create_data(valid_sents)

    # mode
    g = Graph()
    data_size = X.shape[0]
    data_list = list(range(data_size))
    with g.graph.as_default():
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Initialize
            sess.run(tf.global_variables_initializer())
            best_valid_ppl = 100000.
            for epoch in range(1, hp.num_epochs):
                np.random.shuffle(data_list)
                # Train
                train_ppl = 0.
                num_batch = data_size / hp.batch_size
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    sent_ids = data_list[step * hp.batch_size:step * hp.batch_size + hp.batch_size]
                    loss, t_ppl, gs = sess.run([g.optimizer, g.perplexity, g.global_step],
                                            {g.x: X[sent_ids], g.y: Y[sent_ids]})
                    train_ppl += t_ppl
                    if step % 50 == 0:
                        print('\tstep:{} train_ppl:{:.3f}'.format(gs, t_ppl))
                train_ppl /= num_batch
                # Validation
                valid_ppl = 0.
                for idx in range(0, len(valid_X), hp.batch_size):
                    v_ppl = sess.run(g.perplexity, {g.x: valid_X[idx:idx + hp.batch_size],
                                                    g.y: valid_Y[idx:idx + hp.batch_size]})
                    valid_ppl += v_ppl
                valid_ppl /= len(valid_X) / hp.batch_size
                print("[epoch{}] train_PPL={:.3f} validate_PPL={:.3f} ".format(epoch, train_ppl, valid_ppl))
                # Stopping at best valid ppl
                if valid_ppl <= best_valid_ppl * 0.999:
                    best_valid_ppl = valid_ppl
                    saver.save(sess, hp.logdir + '/model.ckpt')
                else:
                    if hp.is_earlystopping:
                        print("Early Stopping...")
                        break
