from keras.optimizers import RMSprop
from keras import Model
import numpy as np


def adding_problem_generator(T: int, batch_size: int = 64):
    """
    A batch generator for the adding problem.
    Code reused from https://github.com/batzner/indrnn/blob/master/examples/addition_rnn.py
    :param batch_size: batch size to use
    :param T: sequence length
    """

    def batch_generator():
        while True:
            """Generate the adding problem dataset"""
            # Build the first sequence
            add_values = np.random.rand(batch_size, T)

            # Build the second sequence with one 1 in each half and 0s otherwise
            add_indices = np.zeros_like(add_values)
            half = int(T / 2)
            for i in range(batch_size):
                first_half = np.random.randint(half)
                second_half = np.random.randint(half, T)
                add_indices[i, [first_half, second_half]] = 1

            # Zip the values and indices in a third dimension:
            # inputs has the shape (batch_size, time_steps, 2)
            inputs = np.dstack((add_values, add_indices))
            targets = np.sum(np.multiply(add_values, add_indices), axis=1)

            yield inputs, targets

    return batch_generator


def fit(model: Model, T: int,
        initial_lr=1e-2,
        final_lr=0,
        epochs=100,
        steps_per_epoch=100,
        batch_size=64,
        validation_steps=10):
    model.compile(optimizer=RMSprop(lr=initial_lr,
                                    decay=(initial_lr - final_lr) / (epochs * steps_per_epoch)),
                  loss='mse')
    gen = adding_problem_generator(T=T, batch_size=batch_size)
    return model.fit_generator(gen(),
                               steps_per_epoch=steps_per_epoch,
                               epochs=epochs,
                               validation_data=gen(),
                               validation_steps=validation_steps,
                               verbose=2)

