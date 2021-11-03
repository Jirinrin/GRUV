from keras.models import Sequential
from keras.layers import TimeDistributed, Dense, Input
from keras.layers.recurrent import LSTM, GRU
from keras import activations, regularizers, initializers, constraints
from keras.engine.input_spec import InputSpec
from keras.engine.base_layer import Layer
import warnings

import keras.backend as K
# def time_distributed_dense(x, w, b=None, dropout=None,
#                            input_dim=None, output_dim=None, timesteps=None):
#     '''Apply y.w + b for every temporal slice y of x.
#     '''
#     if not input_dim:
#         # won't work with TensorFlow
#         input_dim = K.shape(x)[2]
#     if not timesteps:
#         # won't work with TensorFlow
#         timesteps = K.shape(x)[1]
#     if not output_dim:
#         # won't work with TensorFlow
#         output_dim = K.shape(w)[1]

#     if dropout:
#         # apply the same dropout pattern at every timestep
#         ones = K.backend.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
#         dropout_matrix = K.dropout(ones, dropout)
#         expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
#         x *= expanded_dropout_matrix

#     # collapse time dimension and batch dimension together
#     x = K.reshape(x, (-1, input_dim))

#     x = K.dot(x, w)
#     if b:
#         x = x + b
#     # reshape to 3D tensor
#     x = K.reshape(x, (-1, timesteps, output_dim))
#     return x

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	print('tim distr', num_frequency_dimensions, num_hidden_dimensions)
	#This layer converts frequency space to hidden space
    # model.add(Input(input_shape=(None,num_frequency_dimensions)))
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in range(num_recurrent_units):
		model.add(LSTM(num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

def create_gru_network(num_frequency_dimensions, num_hidden_dimensions, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add(TimeDistributedDense(input_dim=num_frequency_dimensions, output_dim=num_hidden_dimensions))
	for cur_unit in range(num_recurrent_units):
		model.add(GRU(num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributedDense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model


class TimeDistributedDense(Layer):
    """Apply a same Dense layer for each dimension[1] (time_dimension) input.

    Especially useful after a recurrent network with 'return_sequence=True'.

    Note: this layer is deprecated, prefer using the `TimeDistributed` wrapper:
    ```python
        model.add(TimeDistributed(Dense(32)))
    ```

    # Input shape
        3D tensor with shape `(nb_sample, time_dimension, input_dim)`.

    # Output shape
        3D tensor with shape `(nb_sample, time_dimension, output_dim)`.

    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
        input_length: length of inputs sequences
            (integer, or None for variable-length sequences).
    """

    def __init__(self, output_dim,
                 init='glorot_uniform',
                 activation=None,
                 weights=None,
                 W_regularizer=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 input_length=None,
                 **kwargs):
        warnings.warn('`TimeDistributedDense` is deprecated, '
                      'And will be removed on May 1st, 2017. '
                      'Please use a `Dense` layer instead.')
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.activation = activations.get(activation)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=3)]
        self.supports_masking = True

        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        super(TimeDistributedDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None,) + input_shape[1:])]
        input_dim = input_shape[2]

        self.W = self.add_weight(shape=(input_dim, self.output_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(self.output_dim,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        # x has shape (samples, timesteps, input_dim)
        input_length = input_shape[1]
        if not input_length:
            if hasattr(K, 'int_shape'):
                input_length = K.int_shape(x)[1]
                if not input_length:
                    raise ValueError('Layer ' + self.name +
                                     ' requires to know the length '
                                     'of its input, but it could not '
                                     'be inferred automatically. '
                                     'Specify it manually by passing '
                                     'an input_shape argument to '
                                     'the first layer in your model.')
            else:
                input_length = K.shape(x)[1]

        # Squash samples and timesteps into a single axis
        x = K.reshape(x, (-1, input_shape[-1]))  # (samples * timesteps, input_dim)
        y = K.dot(x, self.W)  # (samples * timesteps, output_dim)
        if self.bias:
            y += self.b
        # We have to reshape Y to (samples, timesteps, output_dim)
        y = K.reshape(y, (-1, input_length, self.output_dim))  # (samples, timesteps, output_dim)
        y = self.activation(y)
        return y

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim,
                  'input_length': self.input_length}
        base_config = super(TimeDistributedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
