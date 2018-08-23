class Hyperparams:
    word_embed_size = 256
    cnn_layers = 10
    # channel
    filter_size = 1
    filter_h = 5
    filter_w = 5
    block_size = 5
    num_nce_sampled = 1
    batch_size = 128
    vocab_size = 20000
    max_len = 30
    num_epochs = 100
    # optimizing
    lr = 1.0
    momentum = 0.99
    grad_clip = 0.1
    is_earlystopping = True

    # path
    data_path = 'datas/1-billion-word'
    logdir = 'logdir/' + data_path[6:] + '/'
