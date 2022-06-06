from keras.callbacks import LearningRateScheduler
from keras import Model
from keras.optimizers import RMSprop
import numpy as np
import wget
import zipfile
import os


def load_har():
    wget.download('https://github.com/wtaylor17/VPRNN/raw/main/experiments/har/har_data.zip')

    with zipfile.ZipFile('har_data.zip') as zf:
        with zf.open('har_data/x_train.npy', 'r') as fp:
            x_train = np.load(fp)
        with zf.open('har_data/y_train.npy', 'r') as fp:
            y_train = np.load(fp)
        with zf.open('har_data/x_test.npy', 'r') as fp:
            x_test = np.load(fp)
        with zf.open('har_data/y_test.npy', 'r') as fp:
            y_test = np.load(fp)

    os.remove('har_data.zip')

    mu = np.mean(x_train, axis=(0, 1)).reshape((-1, 1, 9))
    std = np.std(x_train, axis=(0, 1)).reshape((-1, 1, 9))
    x_train = (x_train - mu) / std
    x_test = (x_test - mu) / std
    return (x_train, y_train), (x_test, y_test)


def fit(model: Model,
        lr=0.001,
        lr_scheduler=None,
        epochs=300,
        batch_size=100):
    (x_train, y_train), (x_val, y_val) = load_har()

    model.compile(loss='binary_crossentropy',
                  metrics=['acc'],
                  optimizer=RMSprop(lr=lr))
    if lr_scheduler:
        callbacks = [LearningRateScheduler(lr_scheduler)]
    else:
        callbacks = None

    return model.fit(x_train, y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=2,
                     validation_data=[x_val, y_val],
                     callbacks=callbacks)
