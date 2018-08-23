import codecs
import numpy as np
from hyperparams import Hyperparams as hp


def make_dict(sentences, vocab_filepath=hp.data_path + '/1-billion-word.dict'):
    word_dict = {}
    for sentence in sentences:
        for word in sentence.split():
            if word_dict.get(word) is None:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    # write vocab_file
    with codecs.open(vocab_filepath, 'w', 'utf-8') as fout:
        fout.write('<UNK>\t10000000\n')
        for num, word in enumerate(sorted(word_dict.iteritems(), key=lambda d: d[1], reverse=True)):
            if word[1] > 0:
                fout.write(word[0] + '\t' + str(word[1]) + '\n')


def load_sentences(filepath='', is_preprocess=True):
    sentences = codecs.open(filepath, 'r', 'utf-8').read().splitlines()
    # Remove one-word sentences. (includes period)
    if is_preprocess:
        sentences = list(filter(lambda sen: len(sen.split(' ')) > 2, sentences))
    return sentences


def load_vocab():
    vocab = [line.split()[0] for line in
             codecs.open(hp.data_path + '/1-billion-word.dict', 'r', 'utf-8').read().splitlines()][:hp.vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(sentences):
    word2idx, idx2word = load_vocab()

    # Index
    x_list, y_list, sources, targets = [], [], [], []
    for sentence in sentences:
        sentence_list = (sentence + ' <END>').split()
        x = [word2idx.get(word, 1) for word in sentence_list]
        y = x[1:]
        x = x[:-1]
        source = " ".join(sentence_list[:-1])
        target = " ".join(sentence_list[1:])
        if len(x) <= hp.max_len:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            sources.append(source)
            targets.append(target)

    # Pad
    X = np.zeros([len(x_list), hp.max_len], np.int32)
    Y = np.zeros([len(y_list), hp.max_len], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.max_len - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.max_len - len(y)], 'constant', constant_values=(0, 0))
    return X, Y, sources, targets
