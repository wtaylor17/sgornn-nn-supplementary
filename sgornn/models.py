from keras.layers import InputLayer, Dense, RNN, TimeDistributed, Bidirectional, LSTM, SimpleRNNCell
from keras.models import Sequential
from keras.utils.generic_utils import CustomObjectScope

from vprnn.layers import VanillaCell as VPRNNCell

from .layers import ExpRNNCell, FastRNNCell


class FastRNNModel(Sequential):
    def __init__(self, *args,
                 layers=1, dim=128,
                 rots=7, activation='relu',
                 output_dim=1,
                 input_dim=2,
                 output_activation='linear',
                 clip_scalar=True,
                 return_sequences=False,
                 bidirectional=False,
                 cell_class=SimpleRNNCell,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add(InputLayer((None, input_dim)))

        custom_objs = {'VPRNNCell': VPRNNCell,
                       'ExpRNNCell': ExpRNNCell,
                       'FastRNNCell': FastRNNCell}
        for _ in range(layers - 1):
            if bidirectional:
                with CustomObjectScope(custom_objs):
                    if cell_class in [VPRNNCell, ExpRNNCell]:
                        cell = cell_class(dim, n_rotations=rots, activation=activation)
                    else:
                        cell = cell_class(dim, activation=activation)
                    self.add(Bidirectional(RNN(FastRNNCell(cell,
                                                           clip_scalar=clip_scalar),
                                               return_sequences=True)))
            else:
                if cell_class in [VPRNNCell, ExpRNNCell]:
                    cell = cell_class(dim, n_rotations=rots, activation=activation)
                else:
                    cell = cell_class(dim, activation=activation)
                self.add(RNN(FastRNNCell(cell, clip_scalar=clip_scalar),
                             return_sequences=True))
        if bidirectional:
            with CustomObjectScope(custom_objs):
                if cell_class in [VPRNNCell, ExpRNNCell]:
                    cell = cell_class(dim, n_rotations=rots, activation=activation)
                else:
                    cell = cell_class(dim, activation=activation)
                self.add(Bidirectional(RNN(FastRNNCell(cell,
                                                       clip_scalar=clip_scalar),
                                           return_sequences=return_sequences)))
        else:
            if cell_class in [VPRNNCell, ExpRNNCell]:
                cell = cell_class(dim, n_rotations=rots, activation=activation)
            else:
                cell = cell_class(dim, activation=activation)
            self.add(RNN(FastRNNCell(cell, clip_scalar=clip_scalar),
                         return_sequences=return_sequences))

        if return_sequences:
            self.add(TimeDistributed(Dense(output_dim, activation=output_activation)))
        else:
            self.add(Dense(output_dim, activation=output_activation))


class LSTMModel(Sequential):
    def __init__(self, *args,
                 layers=1, dim=128,
                 rots=7, activation='tanh',
                 output_dim=1,
                 input_dim=2,
                 output_activation='linear',
                 clip_scalar=True,
                 return_sequences=False,
                 bidirectional=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add(InputLayer((None, input_dim)))
        for _ in range(layers - 1):
            if bidirectional:
                self.add(Bidirectional(LSTM(dim, return_sequences=True)))
            else:
                self.add(LSTM(dim, return_sequences=True))
        if bidirectional:
            self.add(Bidirectional(LSTM(dim, return_sequences=return_sequences)))
        else:
            self.add(LSTM(dim, return_sequences=return_sequences))
        if return_sequences:
            self.add(TimeDistributed(Dense(output_dim, activation=output_activation)))
        else:
            self.add(Dense(output_dim, activation=output_activation))


class VPRNNModel(Sequential):
    def __init__(self, *args,
                 layers=1, dim=128,
                 rots=7, activation='relu',
                 output_dim=1,
                 input_dim=2,
                 clip_scalar=True,
                 output_activation='linear',
                 return_sequences=False,
                 bidirectional=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add(InputLayer((None, input_dim)))
        custom_objs = {'VPRNNCell': VPRNNCell}
        for _ in range(layers - 1):
            if bidirectional:
                with CustomObjectScope(custom_objs):
                    self.add(Bidirectional(RNN(VPRNNCell(dim,
                                                         n_rotations=rots,
                                                         activation=activation),
                                               return_sequences=True)))
            else:
                self.add(RNN(VPRNNCell(dim,
                                       n_rotations=rots,
                                       activation=activation),
                             return_sequences=True))
        if bidirectional:
            with CustomObjectScope(custom_objs):
                self.add(Bidirectional(RNN(VPRNNCell(dim,
                                                     n_rotations=rots,
                                                     activation=activation),
                                           return_sequences=return_sequences)))
        else:
            self.add(RNN(VPRNNCell(dim,
                                   n_rotations=rots,
                                   activation=activation),
                         return_sequences=return_sequences))
        if return_sequences:
            self.add(TimeDistributed(Dense(output_dim, activation=output_activation)))
        else:
            self.add(Dense(output_dim, activation=output_activation))


class ExpRNNModel(Sequential):
    def __init__(self, *args,
                 layers=1, dim=128,
                 rots=7, activation='relu',
                 output_dim=1,
                 input_dim=2,
                 clip_scalar=True,
                 output_activation='linear',
                 return_sequences=False,
                 bidirectional=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.add(InputLayer((None, input_dim)))
        custom_objs = {'ExpRNNCell': ExpRNNCell}
        for _ in range(layers - 1):
            if bidirectional:
                with CustomObjectScope(custom_objs):
                    self.add(Bidirectional(RNN(ExpRNNCell(dim,
                                                          n_rotations=rots,
                                                          activation=activation),
                                               return_sequences=True)))
            else:
                self.add(RNN(ExpRNNCell(dim,
                                        n_rotations=rots,
                                        activation=activation),
                             return_sequences=True))
        if bidirectional:
            with CustomObjectScope(custom_objs):
                self.add(Bidirectional(RNN(ExpRNNCell(dim,
                                                      n_rotations=rots,
                                                      activation=activation),
                                           return_sequences=return_sequences)))
        else:
            self.add(RNN(ExpRNNCell(dim,
                                    n_rotations=rots,
                                    activation=activation),
                         return_sequences=return_sequences))
        if return_sequences:
            self.add(TimeDistributed(Dense(output_dim, activation=output_activation)))
        else:
            self.add(Dense(output_dim, activation=output_activation))

