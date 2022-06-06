import keras.backend as K
from keras.initializers import Constant
import tensorflow as tf
import keras
from keras.layers import Layer
import numpy as np


class FastRNNCell(Layer):
    def __init__(self, cell: Layer,
                 alpha_init=-3.75,
                 beta_init=3.0,
                 clip_scalar=True,
                 **kwargs):
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.alpha = None
        self.beta = None
        self.clip_scalar = clip_scalar
        self.cell = cell
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.alpha = self.add_weight(name='alpha',
                                     shape=(1, 1),
                                     initializer=Constant(self.alpha_init))
        self.beta = self.add_weight(name='beta',
                                    shape=(1, 1),
                                    initializer=Constant(self.beta_init))
        self.cell.build(input_shape)

    @property
    def state_size(self):
        return self.cell.state_size

    def call(self, inputs, states, **kwargs):
        prev_state = states[0]
        # call the VPRNN cell
        b = K.sigmoid(self.beta)
        a = K.sigmoid(self.alpha)
        if self.clip_scalar:
            b = K.clip(b, 0, 1 - 2 * a)
        h_t, _ = self.cell.call(inputs, states, **kwargs)
        new_state = a * h_t + b * prev_state
        return new_state, [new_state]


class ExpOrthogonal(Layer):
    def __init__(self, units, initializer='henaff', **kwargs):
        self.units = units
        self.kernel = None
        if initializer in ['henaff', 'cayley']:
            # initializers from exprnn paper
            # creates special block diagonals for the skew matrix
            nonzero_entries = [0]
            i = units - 1
            while i > 0 and nonzero_entries[-1] + i < self.units * (self.units - 1) // 2:
                nonzero_entries.append(nonzero_entries[-1] + i)
                i -= 1
            const = np.zeros((self.units * (self.units - 1) // 2,))
            if initializer == 'henaff':
                const[nonzero_entries] = np.random.uniform(-np.pi, np.pi,
                                                           size=(len(nonzero_entries),))
            else:
                u = np.random.uniform(0, np.pi / 2, size=(len(nonzero_entries),))
                const[nonzero_entries] = -np.sqrt((1 - np.cos(u)) / (1 + np.cos(u)))
            self.initializer = keras.initializers.Constant(const)
        else:
            self.initializer = keras.initializers.get(initializer)
        self.skew = None
        self.skew_mat = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # skew symmetric matrix has n*(n-1)/2 params
        self.skew = self.add_weight('skew',
                                    shape=((self.units * (self.units - 1)) // 2,),
                                    initializer=self.initializer)

        # create binary mask with 1s in upper diagonals
        ones = tf.ones((self.units, self.units))
        mask = tf.matrix_band_part(ones, 0, -1) - tf.matrix_band_part(ones, 0, 0)

        # create sparse mat with entries from skew and convert to dense
        skew_mat = tf.SparseTensor(tf.where(tf.not_equal(mask, 0)),
                                   self.skew, (self.units, self.units))
        skew_mat_dense = tf.sparse.to_dense(skew_mat)

        # compute skew symmetric and exp
        self.skew_mat = skew_mat_dense - tf.transpose(skew_mat_dense)
        self.kernel = tf.linalg.expm(self.skew_mat)

    def call(self, x, **kwargs):
        return tf.matmul(x, self.kernel)


class ExpRNNCell(Layer):
    def __init__(self, units,
                 orthogonal_init='uniform',
                 dense_init='uniform',
                 activation='relu',
                 n_rotations=None,
                 **kwargs):
        self.orthogonal_init = orthogonal_init
        self.dense_init = dense_init
        self.dense = None
        self.orthogonal = None
        self.units = units
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        super().build(input_shape)
        self.dense = keras.layers.Dense(self.units, activation=None,
                                        kernel_initializer=self.dense_init)
        self.orthogonal = ExpOrthogonal(self.units,
                                        initializer=self.orthogonal_init)
        self.dense.build(input_shape)
        self.orthogonal.build((self.units,))
        self.built = True

    def call(self, inputs, states):
        wx_b = self.dense(inputs)
        uh = self.orthogonal(states[0])
        h = self.activation(wx_b + uh)
        return h, [h]
