import keras.backend as K
from keras import activations
from keras.engine.topology import Layer, InputSpec


class FMLayer(Layer):
    def __init__(self, feature_num,
    			feature_size,
                 embedding_size,
                 output_dim=1,
                 activation=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(FMLayer, self).__init__(**kwargs)

        self.output_dim = output_dim
        self.embedding_size = embedding_size
        self.activation = activations.get(activation)
        self.input_spec = InputSpec(ndim=2)
        self.feature_num = feature_num
        self.feature_size = feature_size

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        numeric_size = input_dim - self.feature_num
        self.numeric_size = numeric_size
        all_size = numeric_size + self.feature_size

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.w_one_hot = self.add_weight(name='one_one_hot', 
                                 shape=(self.feature_size, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.w_numeric = self.add_weight(name='one_numeric', 
                                 shape=(numeric_size, self.output_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v_one_hot = self.add_weight(name='two_one_hot', 
                                 shape=(self.feature_size, self.embedding_size),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.v_numeric = self.add_weight(name='two_numeric', 
                                 shape=(numeric_size, self.embedding_size),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='bias', 
                                 shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

        super(FMLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        one_hot_feature_index = K.cast(K.slice(inputs, (0, 0), (-1, self.feature_num)), "int32")
        numeric_feature = K.slice(inputs, (0, self.feature_num), (-1, -1))

        ## first order
        first_order_index = K.reshape(one_hot_feature_index, (-1,))
        get_first_order_weights = K.gather(self.w_one_hot, first_order_index)
        first_order_weights = K.reshape(get_first_order_weights, (-1, self.feature_num))

        first_order = K.sum(first_order_weights, 1) + K.sum(K.dot(numeric_feature, self.w_numeric), 1)

        ## second order
        get_second_order_weights = K.gather(self.v_one_hot, first_order_index)
        second_order_weights = K.reshape(get_second_order_weights, (-1, self.feature_num, self.embedding_size))
        numeric_weights = K.expand_dims(self.v_numeric, 0) * K.expand_dims(numeric_feature, -1)

        all_weights = K.concatenate([second_order_weights, numeric_weights], axis=1)
        weights_sum_square = K.sum(K.square(all_weights), 1)
        weights_square_sum = K.square(K.sum(all_weights, 1))
        second_order = 0.5*K.sum(weights_square_sum - weights_sum_square, 1)

        output = first_order + second_order + self.b

        if self.activation is not None:
        	output = self.activation(output)
        output = K.expand_dims(output, -1)
        return output



        '''X_square = K.square(inputs)

        xv = K.square(K.dot(inputs, self.v))
        xw = K.dot(inputs, self.w)

        p = 0.5 * K.sum(xv - K.dot(X_square, K.square(self.v)), 1)
        rp = K.repeat_elements(K.reshape(p, (-1, 1)), self.output_dim, axis=-1)

        f = xw + rp + self.b

        output = K.reshape(f, (-1, self.output_dim))
        
        if self.activation is not None:
            output = self.activation(output)

        return output'''

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.output_dim




