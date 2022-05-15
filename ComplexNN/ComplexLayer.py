import tensorflow as tf
from tensorflow.keras.layers import Activation,ZeroPadding1D,Conv1D,Dense
from tensorflow.python.keras import backend
from tensorflow import nn
from tensorflow.python.ops import array_ops

import numpy as np
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints
import keras.backend as K


''' DIY '''
def real_convert_to_complex(x):
    # shape=(batch_size,L,2)
    x_real = tf.slice(x,[0,0,0],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    x_imag = tf.slice(x,[0,0,tf.shape(x)[-1]//2],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    xc = tf.complex(x_real,x_imag)
    # shape=(batch_size,L,1)
    return xc

def complex_convert_to_real(xc):
    # shape=(batch_size,L,m)
    x_real = tf.math.real(xc)
    x_imag = tf.math.imag(xc)
    x = tf.concat([x_real,x_imag],axis=-1)
    # shape=(batch_size,L,2)
    return x

'''Support Complex Number input'''
'''Complex Convolutional'''
def single_complex_conv1d(inputs,filters):
    # inputs shape=(batch_size,kernel_size,input_dim) complex
    # filters shape=(kernel_size,input_dim,filters) complex
    filters = tf.expand_dims(filters, axis=0)
    # kernel  (1,kernel_size,input_dim,filters)
    inputs = tf.expand_dims(inputs,axis=-1)
    #  (batch_size,kernel_size,input_dim,1)
    output = tf.reduce_sum(tf.reduce_sum(tf.multiply(filters, inputs), axis=1), axis=1) # (batch_size,kernel_size,input_dim,filters)
    # (batch_size,filters)
    return output

def bias_add(inputs,bias):
    return inputs+bias


class Complex_Conv1D(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,padding='valid',activation='linear'):
        super(Complex_Conv1D,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
    def build(self,input_shape):
        input_dim = input_shape[-1]
        self.kernel_real = self.add_weight(name='kernel_real',
                                            shape=(self.kernel_size,input_dim,self.filters),
                                            initializer='glorot_uniform',
                                            trainable=True)
        self.kernel_imag = self.add_weight(name='kernel_imag',
                                            shape=(self.kernel_size,input_dim,self.filters),
                                            initializer='glorot_uniform',
                                            trainable=True)
        self.bias_real = self.add_weight(name='bias_real',
                                          shape=(self.filters,),
                                          initializer='zeros',
                                          trainable=True)
        self.bias_imag = self.add_weight(name='bias_imag',
                                          shape=(self.filters,),
                                          initializer='zeros',
                                          trainable=True)
        self.act = Activation(self.activation)
    def call(self,xc):
        wc = tf.complex(self.kernel_real,self.kernel_imag)
        bc = tf.complex(self.bias_real,self.bias_imag)

        if self.padding == 'valid':
            outputs_side = (tf.shape(xc)[1] - self.kernel_size) // self.strides + 1
            # kernel  (1,kernel_size,input_dim,filters)
        else: # padding='same'
            outputs_side = tf.shape(xc)[1]
            padding_side = ((tf.shape(xc)[1] - 1) * self.strides - tf.shape(xc)[1] + self.filters) // 2
            x_real = tf.math.real(xc)  # (batch_size,L,dim)
            x_imag = tf.math.imag(xc)  # (batch_size,L,dim)
            x = tf.concat([x_real, x_imag], axis=-1)
            x_Pad = ZeroPadding1D(padding=int(padding_side))(x)
            x_Pad_real = tf.slice(x_Pad, [0, 0, 0], [tf.shape(x_Pad)[0], tf.shape(x_Pad)[1], tf.shape(x_Pad)[-1] // 2])
            x_Pad_imag = tf.slice(x_Pad, [0, 0, tf.shape(x_Pad)[-1] // 2],
                                  [tf.shape(x_Pad)[0], tf.shape(x_Pad)[1], tf.shape(x_Pad)[-1] // 2])
            xc= tf.complex(x_Pad_real, x_Pad_imag)

        yc = []
        for i in range(outputs_side):
            xctmp = tf.slice(xc, [0, i * self.strides, 0], [tf.shape(xc)[0], self.kernel_size, tf.shape(xc)[-1]])
            # (batch_size,kernel_size,input_dim)
            yctmp = single_complex_conv1d(xctmp, wc)
            # (bacth_size,filters)
            yctmp = tf.expand_dims(yctmp,axis=1)
            if i == 0:
                yc = yctmp
            else:
                yc = tf.concat([yc,yctmp],axis=1)
                # (batch_size,output_side,filters)

        # shape=(batch_size,L,filters)
        outc = bias_add(yc,bc)
        outc = self.act(outc)
        return outc
    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'activation':
                self.activation
        }
        base_config = super(Complex_Conv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''Complex Dense'''
class Complex_Dense(tf.keras.layers.Layer):
    def __init__(self,units,activation):
        super(Complex_Dense,self).__init__()
        self.units = units
        self.activation = activation
    def build(self,input_shape):
        input_dim = input_shape[-1]
        self.weight_real = self.add_weight(name='weight_real',
                                            shape=(input_dim,self.units),
                                            initializer='glorot_uniform',
                                            trainable=True)
        self.weight_imag = self.add_weight(name='weight_imag',
                                           shape=(input_dim, self.units),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.bias_real = self.add_weight(name='bias_real',
                                         shape=(self.units,),
                                         initializer='zeros',
                                         trainable=True)
        self.bias_imag = self.add_weight(name='bias_imag',
                                         shape=(self.units,),
                                         initializer='zeros',
                                         trainable=True)
        self.act = Activation(self.activation)
    def call(self,xc):
        # (batch_size,L,dim)
        wc = tf.complex(self.kernel_real, self.kernel_imag)
        bc = tf.complex(self.bias_real, self.bias_imag)

        yc = tf.matmul(xc,wc)
        outc = bias_add(yc, bc)
        return outc
    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                self.activation
        }
        base_config = super(Complex_Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



'''Supprot Real Number input'''
'''Complex LSTM'''
# tensorflow 中的RNN后端就是用for循环实现的
class ComplexLSTMCell(tf.keras.layers.Layer):
    def __init__(self,units):
        super(ComplexLSTMCell,self).__init__()
        self.units = units 
        self.state_size = [2*self.units, 2*self.units]
    def build(self,input_shape):
        self.kernel_real = self.add_weight(shape=(input_shape[-1]//2, 4*self.units),
                                      initializer='glorot_uniform',
                                      name='kernel_real')
        self.kernel_imag = self.add_weight(shape=(input_shape[-1]//2, 4*self.units),
                                           initializer='glorot_uniform',
                                           name='kernel_imag')
        self.recurrent_kernel_real = self.add_weight(shape=(self.units, 4*self.units),
                                        initializer='glorot_uniform',
                                        name='recurrent_kernel_real')
        self.recurrent_kernel_imag = self.add_weight(shape=(self.units, 4*self.units),
                                                     initializer='glorot_uniform',
                                                     name='recurrent_kernel_imag')
        self.bias_real = self.add_weight(shape=(4*self.units,),
                                      initializer='zeros',
                                      name='bias_real')
        self.bias_imag = self.add_weight(shape=(4*self.units,),
                                         initializer='zeros',
                                         name='bias_imag')

    def split_real_imag(self,x):
        x_real = tf.slice(x, [0, 0], [tf.shape(x)[0], tf.shape(x)[-1] // 2])
        x_imag = tf.slice(x, [0, tf.shape(x)[-1] // 2], [tf.shape(x)[0], tf.shape(x)[-1] // 2])
        return x_real, x_imag

    def call(self,cell_inputs,cell_states):
        # cell inputs (batch_size,dim)
        h_tm1 = cell_states[0]  # previous memory state
        c_tm1 = cell_states[1]  # previous carry state

        h_tm1_real, h_tm1_imag = self.split_real_imag(h_tm1) # (batch_size,units)
        c_tm1_real, c_tm1_imag = self.split_real_imag(c_tm1)

        cell_inputs_real, cell_inputs_imag = self.split_real_imag(cell_inputs) # (batch_size,dim//2)
        '''Compute Real'''
        z_real = backend.dot(cell_inputs_real, self.kernel_real) - backend.dot(cell_inputs_imag, self.kernel_imag)
        z_real += backend.dot(h_tm1_real, self.recurrent_kernel_real) - backend.dot(h_tm1_imag,self.recurrent_kernel_imag)
        z_real = bias_add(z_real, self.bias_real)
        # (batch_size,4*units)

        z0_real, z1_real, z2_real, z3_real = array_ops.split(z_real, 4, axis=1)
        # (batch_size,units)
        '''Compute Imag'''
        z_imag = backend.dot(cell_inputs_real, self.kernel_imag) + backend.dot(cell_inputs_imag, self.kernel_real)
        z_imag += backend.dot(h_tm1_real, self.recurrent_kernel_imag) + backend.dot(h_tm1_imag,self.recurrent_kernel_real)
        z_imag = bias_add(z_imag, self.bias_imag)
        # (batch_size,4*units)

        z0_imag, z1_imag, z2_imag, z3_imag = array_ops.split(z_imag, 4, axis=1)
        # (batch_size,units)

        i_real = nn.sigmoid(z0_real)  # (batch_size,units)
        f_real = nn.sigmoid(z1_real)
        i_imag = nn.sigmoid(z0_imag)
        f_imag = nn.sigmoid(z1_imag)

        c_real = (f_real * c_tm1_real - f_imag * c_tm1_imag) + (
                i_real * nn.tanh(z2_real) - i_imag * nn.tanh(z2_imag))
        c_imag = (f_real * c_tm1_imag + f_imag * c_tm1_real) + (
                i_real * nn.tanh(z2_imag) - i_imag * nn.tanh(z2_real))
        # (batch_size,units)

        o_real = nn.sigmoid(z3_real)
        o_imag = nn.sigmoid(z3_imag)

        h_real = o_real * nn.tanh(c_real) - o_imag * nn.tanh(c_imag)
        h_imag = o_real * nn.tanh(c_imag) - o_imag * nn.tanh(c_real)

        h = tf.concat([h_real, h_imag], axis=-1)
        c = tf.concat([c_real, c_imag], axis=-1)
        # (batch_size,2*units)
        return h, [h,c]


    def get_config(self):
        config = {
            'units':
                self.units,
        }
        base_config = super(ComplexLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''Complex Convolutional'''
class ComplexConv1D(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,padding='valid',activation='linear'):
        super(ComplexConv1D,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
    def build(self,input_shape):
        self.conv_real = Conv1D(filters=self.filters,kernel_size=self.kernel_size,
                                padding=self.padding,use_bias=False)

        self.conv_imag = Conv1D(filters=self.filters,kernel_size=self.kernel_size,
                                padding=self.padding,use_bias=False)

        self.bias_real = self.add_weight(name='bias_real',
                                         shape=(self.filters,),
                                         initializer='zeros',
                                         trainable=True)

        self.bias_imag = self.add_weight(name='bias_imag',
                                         shape=(self.filters,),
                                         initializer='zeros',
                                         trainable=True)
        self.act = Activation(self.activation)
    def call(self,x):
        x_real = tf.slice(x,[0,0,0],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
        x_imag = tf.slice(x,[0,0,tf.shape(x)[-1]//2],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])

        y_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        y_imag = self.conv_real(x_imag) + self.conv_imag(x_real)

        out_real = bias_add(y_real,self.bias_real)
        out_imag = bias_add(y_imag,self.bias_imag)
        out = tf.concat([out_real,out_imag],axis=-1)

        out = self.act(out)
        return out
    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'activation':
                self.activation
        }
        base_config = super(ComplexConv1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
'''Complex Dense'''
class ComplexDense(tf.keras.layers.Layer):
    def __init__(self, units,activation='linear'):
        super(ComplexDense, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        self.kernel_real = Dense(units=self.units,use_bias=False)

        self.kernel_imag = Dense(units=self.units,use_bias=False)

        self.bias_real = self.add_weight(name='bias_real',
                                         shape=(self.units,),
                                         initializer='zeros',
                                         trainable=True)

        self.bias_imag = self.add_weight(name='bias_imag',
                                         shape=(self.units,),
                                         initializer='zeros',
                                         trainable=True)
        self.act = Activation(self.activation)

    def call(self, x):
        x_real = tf.slice(x, [0, 0, 0], [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1] // 2])
        x_imag = tf.slice(x, [0, 0, tf.shape(x)[-1] // 2], [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[-1] // 2])

        y_real = self.kernel_real(x_real) - self.kernel_imag(x_imag)
        y_imag = self.kernel_real(x_imag) + self.kernel_imag(x_real)

        out_real = bias_add(y_real, self.bias_real)
        out_imag = bias_add(y_imag, self.bias_imag)
        out = tf.concat([out_real, out_imag], axis=-1)

        out = self.act(out)
        return out
    def get_config(self):
        config = {
            'units':
                self.units,
            'activation':
                self.activation
        }
        base_config = super(ComplexDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



'''Complex BatchNormalization'''
def sqrt_init(shape, dtype=None):
    value = K.ones(shape) / tf.sqrt(2.0)
    return value

def sanitizedInitGet(init):
    if init in ["sqrt_init"]:
        return sqrt_init
    else:
        return initializers.get(init)

def sanitizedInitSer(init):
    if init in [sqrt_init]:
        return "sqrt_init"
    else:
        return initializers.serialize(init)


def complex_standardization(input_centred, Vrr, Vii, Vri,layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    variances_broadcast = [1] * ndim
    variances_broadcast[axis] = input_dim
    if layernorm:
        variances_broadcast[0] = K.shape(input_centred)[0]

    # We require the covariance matrix's inverse square root. That first requires
    # square rooting, followed by inversion (I do this in that order because during
    # the computation of square root we compute the determinant we'll need for
    # inversion as well).

    # tau = Vrr + Vii = Trace. Guaranteed >= 0 because SPD
    tau = Vrr + Vii
    # delta = (Vrr * Vii) - (Vri ** 2) = Determinant. Guaranteed >= 0 because SPD
    delta = (Vrr * Vii) - (Vri ** 2)

    s = K.sqrt(delta)  # Determinant of square root matrix
    t = K.sqrt(tau + 2 * s)

    # The square root matrix could now be explicitly formed as
    #       [ Vrr+s Vri   ]
    # (1/t) [ Vir   Vii+s ]
    # https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    # but we don't need to do this immediately since we can also simultaneously
    # invert. We can do this because we've already computed the determinant of
    # the square root matrix, and can thus invert it using the analytical
    # solution for 2x2 matrices
    #      [ A B ]             [  D  -B ]
    # inv( [ C D ] ) = (1/det) [ -C   A ]
    # http://mathworld.wolfram.com/MatrixInverse.html
    # Thus giving us
    #           [  Vii+s  -Vri   ]
    # (1/s)(1/t)[ -Vir     Vrr+s ]
    # So we proceed as follows:

    inverse_st = 1.0 / (s * t)
    Wrr = (Vii + s) * inverse_st
    Wii = (Vrr + s) * inverse_st
    Wri = -Vri * inverse_st

    # And we have computed the inverse square root matrix W = sqrt(V)!
    # Normalization. We multiply, x_normalized = W.x.

    # The returned result will be a complex standardized input
    # where the real and imaginary parts are obtained as follows:
    # x_real_normed = Wrr * x_real_centred + Wri * x_imag_centred
    # x_imag_normed = Wri * x_real_centred + Wii * x_imag_centred

    broadcast_Wrr = K.reshape(Wrr, variances_broadcast)
    broadcast_Wri = K.reshape(Wri, variances_broadcast)
    broadcast_Wii = K.reshape(Wii, variances_broadcast)

    cat_W_4_real = K.concatenate([broadcast_Wrr, broadcast_Wii], axis=axis)
    cat_W_4_imag = K.concatenate([broadcast_Wri, broadcast_Wri], axis=axis)

    if (axis == 1 and ndim != 3) or ndim == 2:
        centred_real = input_centred[:, :input_dim]
        centred_imag = input_centred[:, input_dim:]
    elif ndim == 3:
        centred_real = input_centred[:, :, :input_dim]
        centred_imag = input_centred[:, :, input_dim:]
    elif axis == -1 and ndim == 4:
        centred_real = input_centred[:, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, input_dim:]
    elif axis == -1 and ndim == 5:
        centred_real = input_centred[:, :, :, :, :input_dim]
        centred_imag = input_centred[:, :, :, :, input_dim:]
    else:
        raise ValueError(
            'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
            'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
        )
    rolled_input = K.concatenate([centred_imag, centred_real], axis=axis)

    output = cat_W_4_real * input_centred + cat_W_4_imag * rolled_input

    #   Wrr * x_real_centered | Wii * x_imag_centered
    # + Wri * x_imag_centered | Wri * x_real_centered
    # -----------------------------------------------
    # = output

    return output


def ComplexBN(input_centred, Vrr, Vii, Vri, beta,gamma_rr, gamma_ri, gamma_ii, scale=True,
              center=True, layernorm=False, axis=-1):
    ndim = K.ndim(input_centred)
    input_dim = K.shape(input_centred)[axis] // 2
    if scale:
        gamma_broadcast_shape = [1] * ndim
        gamma_broadcast_shape[axis] = input_dim
    if center:
        broadcast_beta_shape = [1] * ndim
        broadcast_beta_shape[axis] = input_dim * 2

    if scale:
        standardized_output = complex_standardization(
            input_centred, Vrr, Vii, Vri,
            layernorm,
            axis=axis
        )

        # Now we perform th scaling and Shifting of the normalized x using
        # the scaling parameter
        #           [  gamma_rr gamma_ri  ]
        #   Gamma = [  gamma_ri gamma_ii  ]
        # and the shifting parameter
        #    Beta = [beta_real beta_imag].T
        # where:
        # x_real_BN = gamma_rr * x_real_normed + gamma_ri * x_imag_normed + beta_real
        # x_imag_BN = gamma_ri * x_real_normed + gamma_ii * x_imag_normed + beta_imag

        broadcast_gamma_rr = K.reshape(gamma_rr, gamma_broadcast_shape)
        broadcast_gamma_ri = K.reshape(gamma_ri, gamma_broadcast_shape)
        broadcast_gamma_ii = K.reshape(gamma_ii, gamma_broadcast_shape)

        cat_gamma_4_real = K.concatenate([broadcast_gamma_rr, broadcast_gamma_ii], axis=axis)
        cat_gamma_4_imag = K.concatenate([broadcast_gamma_ri, broadcast_gamma_ri], axis=axis)
        if (axis == 1 and ndim != 3) or ndim == 2:
            centred_real = standardized_output[:, :input_dim]
            centred_imag = standardized_output[:, input_dim:]
        elif ndim == 3:
            centred_real = standardized_output[:, :, :input_dim]
            centred_imag = standardized_output[:, :, input_dim:]
        elif axis == -1 and ndim == 4:
            centred_real = standardized_output[:, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, input_dim:]
        elif axis == -1 and ndim == 5:
            centred_real = standardized_output[:, :, :, :, :input_dim]
            centred_imag = standardized_output[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(axis) + '; ndim: ' + str(ndim) + '.'
            )
        rolled_standardized_output = K.concatenate([centred_imag, centred_real], axis=axis)
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output + broadcast_beta
        else:
            return cat_gamma_4_real * standardized_output + cat_gamma_4_imag * rolled_standardized_output
    else:
        if center:
            broadcast_beta = K.reshape(beta, broadcast_beta_shape)
            return input_centred + broadcast_beta
        else:
            return input_centred

class ComplexBatchNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.9,
                 epsilon=1e-4,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_diag_initializer='sqrt_init',
                 gamma_off_initializer='zeros',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='sqrt_init',
                 moving_covariance_initializer='zeros',
                 beta_regularizer=None,
                 gamma_diag_regularizer=None,
                 gamma_off_regularizer=None,
                 beta_constraint=None,
                 gamma_diag_constraint=None,
                 gamma_off_constraint=None,
                 **kwargs):
        super(ComplexBatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = sanitizedInitGet(beta_initializer)
        self.gamma_diag_initializer = sanitizedInitGet(gamma_diag_initializer)
        self.gamma_off_initializer = sanitizedInitGet(gamma_off_initializer)
        self.moving_mean_initializer = sanitizedInitGet(moving_mean_initializer)
        self.moving_variance_initializer = sanitizedInitGet(moving_variance_initializer)
        self.moving_covariance_initializer = sanitizedInitGet(moving_covariance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_diag_regularizer = regularizers.get(gamma_diag_regularizer)
        self.gamma_off_regularizer = regularizers.get(gamma_off_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_diag_constraint = constraints.get(gamma_diag_constraint)
        self.gamma_off_constraint = constraints.get(gamma_off_constraint)

    def build(self, input_shape):

        ndim = len(input_shape)

        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})

        param_shape = (input_shape[self.axis] // 2,)

        if self.scale:
            self.gamma_rr = self.add_weight(shape=param_shape,
                                            name='gamma_rr',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ii = self.add_weight(shape=param_shape,
                                            name='gamma_ii',
                                            initializer=self.gamma_diag_initializer,
                                            regularizer=self.gamma_diag_regularizer,
                                            constraint=self.gamma_diag_constraint)
            self.gamma_ri = self.add_weight(shape=param_shape,
                                            name='gamma_ri',
                                            initializer=self.gamma_off_initializer,
                                            regularizer=self.gamma_off_regularizer,
                                            constraint=self.gamma_off_constraint)
            self.moving_Vrr = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vrr',
                                              trainable=False)
            self.moving_Vii = self.add_weight(shape=param_shape,
                                              initializer=self.moving_variance_initializer,
                                              name='moving_Vii',
                                              trainable=False)
            self.moving_Vri = self.add_weight(shape=param_shape,
                                              initializer=self.moving_covariance_initializer,
                                              name='moving_Vri',
                                              trainable=False)
        else:
            self.gamma_rr = None
            self.gamma_ii = None
            self.gamma_ri = None
            self.moving_Vrr = None
            self.moving_Vii = None
            self.moving_Vri = None

        if self.center:
            self.beta = self.add_weight(shape=(input_shape[self.axis],),
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
            self.moving_mean = self.add_weight(shape=(input_shape[self.axis],),
                                               initializer=self.moving_mean_initializer,
                                               name='moving_mean',
                                               trainable=False)
        else:
            self.beta = None
            self.moving_mean = None

        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        ndim = len(input_shape)
        reduction_axes = list(range(ndim))
        del reduction_axes[self.axis]
        input_dim = input_shape[self.axis] // 2
        mu = K.mean(inputs, axis=reduction_axes)
        broadcast_mu_shape = [1] * len(input_shape)
        broadcast_mu_shape[self.axis] = input_shape[self.axis]
        broadcast_mu = K.reshape(mu, broadcast_mu_shape)
        if self.center:
            input_centred = inputs - broadcast_mu
        else:
            input_centred = inputs
        centred_squared = input_centred ** 2
        if (self.axis == 1 and ndim != 3) or ndim == 2:
            centred_squared_real = centred_squared[:, :input_dim]
            centred_squared_imag = centred_squared[:, input_dim:]
            centred_real = input_centred[:, :input_dim]
            centred_imag = input_centred[:, input_dim:]
        elif ndim == 3:
            centred_squared_real = centred_squared[:, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, input_dim:]
            centred_real = input_centred[:, :, :input_dim]
            centred_imag = input_centred[:, :, input_dim:]
        elif self.axis == -1 and ndim == 4:
            centred_squared_real = centred_squared[:, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, input_dim:]
        elif self.axis == -1 and ndim == 5:
            centred_squared_real = centred_squared[:, :, :, :, :input_dim]
            centred_squared_imag = centred_squared[:, :, :, :, input_dim:]
            centred_real = input_centred[:, :, :, :, :input_dim]
            centred_imag = input_centred[:, :, :, :, input_dim:]
        else:
            raise ValueError(
                'Incorrect Batchnorm combination of axis and dimensions. axis should be either 1 or -1. '
                'axis: ' + str(self.axis) + '; ndim: ' + str(ndim) + '.'
            )
        if self.scale:
            Vrr = K.mean(
                centred_squared_real,
                axis=reduction_axes
            ) + self.epsilon
            Vii = K.mean(
                centred_squared_imag,
                axis=reduction_axes
            ) + self.epsilon
            # Vri contains the real and imaginary covariance for each feature map.
            Vri = K.mean(
                centred_real * centred_imag,
                axis=reduction_axes,
            )
        elif self.center:
            Vrr = None
            Vii = None
            Vri = None
        else:
            raise ValueError('Error. Both scale and center in batchnorm are set to False.')

        input_bn = ComplexBN(
            input_centred, Vrr, Vii, Vri,
            self.beta, self.gamma_rr, self.gamma_ri,
            self.gamma_ii, self.scale, self.center,
            axis=self.axis
        )
        if training in {0, False}:
            return input_bn
        else:
            update_list = []
            if self.center:
                update_list.append(K.moving_average_update(self.moving_mean, mu, self.momentum))
            if self.scale:
                update_list.append(K.moving_average_update(self.moving_Vrr, Vrr, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vii, Vii, self.momentum))
                update_list.append(K.moving_average_update(self.moving_Vri, Vri, self.momentum))
            self.add_update(update_list, inputs)

            def normalize_inference():
                if self.center:
                    inference_centred = inputs - K.reshape(self.moving_mean, broadcast_mu_shape)
                else:
                    inference_centred = inputs
                return ComplexBN(
                    inference_centred, self.moving_Vrr, self.moving_Vii,
                    self.moving_Vri, self.beta, self.gamma_rr, self.gamma_ri,
                    self.gamma_ii, self.scale, self.center, axis=self.axis
                )

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(input_bn,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': sanitizedInitSer(self.beta_initializer),
            'gamma_diag_initializer': sanitizedInitSer(self.gamma_diag_initializer),
            'gamma_off_initializer': sanitizedInitSer(self.gamma_off_initializer),
            'moving_mean_initializer': sanitizedInitSer(self.moving_mean_initializer),
            'moving_variance_initializer': sanitizedInitSer(self.moving_variance_initializer),
            'moving_covariance_initializer': sanitizedInitSer(self.moving_covariance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_diag_regularizer': regularizers.serialize(self.gamma_diag_regularizer),
            'gamma_off_regularizer': regularizers.serialize(self.gamma_off_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_diag_constraint': constraints.serialize(self.gamma_diag_constraint),
            'gamma_off_constraint': constraints.serialize(self.gamma_off_constraint),
        }
        base_config = super(ComplexBatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
