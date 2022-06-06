import numpy as np
from keras.preprocessing.text import Tokenizer
from keras import Model
from keras.optimizers import RMSprop
import keras.backend as K
import io
import os
import wget
import zipfile


def ptb_supervised(raw_data, num_steps=300):
    raw_data = np.array(raw_data, dtype=np.int32)
    data_len = len(raw_data)

    X, Y = [], []
    for i in range(num_steps, data_len - 1, num_steps):
        X.append(raw_data[i - num_steps:i])
        Y.append(raw_data[i - num_steps + 1:i + 1])
    return np.array(X), np.array(Y)


def batchify_inds(N, batch_size):
    n_batches = N // batch_size
    batch_inds = [[] for _ in range(n_batches)]
    for i in range(batch_size * n_batches):
        batch_inds[i % n_batches].append(i)
    return batch_inds


def load_sequences():
    wget.download('https://data.deepai.org/ptbdataset.zip')

    with zipfile.ZipFile('ptbdataset.zip') as zf:
        with io.TextIOWrapper(zf.open('ptb.train.txt', 'r'), encoding="utf-8") as fp:
            train_raw = fp.read().replace('\n', '<eos>')
        with io.TextIOWrapper(zf.open('ptb.test.txt', 'r'), encoding="utf-8") as fp:
            test_raw = fp.read().replace('\n', '<eos>')
        with io.TextIOWrapper(zf.open('ptb.valid.txt', 'r'), encoding="utf-8") as fp:
            valid_raw = fp.read().replace('\n', '<eos>')
    os.remove('ptbdataset.zip')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([train_raw])

    train_seq, valid_seq, test_seq = tokenizer.texts_to_sequences([train_raw,
                                                                   valid_raw,
                                                                   test_raw])
    return train_seq, valid_seq, test_seq


def load_ptb_data(num_steps=300):
    train, valid, test = load_sequences()

    return (ptb_supervised(train, num_steps=num_steps),
            ptb_supervised(valid, num_steps=num_steps),
            ptb_supervised(test, num_steps=num_steps))


def entropy(y_true, y_pred):
    # y_true has shape (batch, steps, 1)
    # y_pred has shape (batch, steps, vocab)
    vocab_size = K.int_shape(y_pred)[-1]
    y_true = K.reshape(y_true, (-1,))
    y_pred = K.reshape(y_pred, (-1, vocab_size))
    return K.mean(K.sparse_categorical_crossentropy(y_true, y_pred))


def fit(model: Model,
        lr=0.001,
        epochs=300,
        batch_size=100,
        num_steps=300,
        decay_after=199):
    train, valid, test = load_ptb_data(num_steps=num_steps)

    model.compile(loss=entropy,
                  optimizer=RMSprop(lr=lr))

    batch_inds = batchify_inds(len(train[0]), batch_size)
    x_train = np.concatenate([train[0][b] for b in batch_inds], axis=0)
    y_train = np.concatenate([train[1][b] for b in batch_inds], axis=0)

    batch_inds = batchify_inds(len(valid[0]), batch_size)
    x_val = np.concatenate([valid[0][b] for b in batch_inds], axis=0)
    y_val = np.concatenate([valid[1][b] for b in batch_inds], axis=0)

    for e in range(epochs):
        model.fit(x_train, y_train,
                  epochs=1, verbose=2,
                  validation_data=[x_val, y_val],
                  batch_size=batch_size,
                  shuffle=False)
        if decay_after and e >= decay_after:
            model.optimizer.learning_rate.assign(model.optimizer.learning_rate * .1)
        print(f'EPOCH {e+1}/{epochs}: {evaluate(model, num_steps=num_steps, batch_size=batch_size)}')
        model.reset_states()


def evaluate(model: Model,
             num_steps=300, batch_size=100):
    train, valid, test = load_ptb_data(num_steps=num_steps)

    # get the entropy average
    model.reset_states()
    batch_inds = batchify_inds(len(test[0]), batch_size)
    x = np.concatenate([test[0][b] for b in batch_inds], axis=0)
    y = np.concatenate([test[1][b] for b in batch_inds], axis=0)
    xe = model.evaluate(x, y, batch_size=batch_size)
    if type(xe) is list:
        xe = xe[0]
    # calculate perplexity
    ppl = float(np.exp(xe))
    return xe, ppl
