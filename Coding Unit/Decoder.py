import tensorflow as tf
from tensorflow.keras.layers import Conv1D

from DL_Communication_System.Coding_Unit.Iteration_dec import Iteration_dec_CNN_SBI,Iteration_dec_CNN_PRI,Iteration_dec_RNN_SBI,Iteration_dec_RNN_PRI
from DL_Communication_System.Coding_Unit.Modulation import DeModulator

'''CNN'''
class Decoder_CNN_PRI(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim,k,n):
        super(Decoder_CNN_PRI,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.iter = Iteration_dec_CNN_PRI(batch_size,L,dim)
        self.demodulation = DeModulator(k,n,L,dim)
        self.dec_out = Conv1D(filters=2,strides=1,kernel_size=1,activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,1)),
                             trainable=True)
    def call(self,x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [tf.shape(x)[0], tf.shape(x)[1], 1])
        de2 = tf.slice(d, [0, 0, 1],[tf.shape(x)[0],tf.shape(x)[1],1])
        dI = tf.slice(d, [0, 0, 2],[tf.shape(x)[0],tf.shape(x)[1],1])
        # shape=(batch_size,L,1)
        p = self.p
        for i in range(5):
            p = self.iter(de1, de2, dI,p)
        # shape=(batch_size,L,1)
        y = self.dec_out(p)
        return y
    def get_config(self):
        config = {
            'batch_size':
                self.batch_size,
            'L':
                self.L,
            'dim':
                self.dim,
            'k':
                self.k,
            'n':
                self.n
        }
        base_config = super(Decoder_CNN_PRI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Decoder_CNN_SBI(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim,k,n,B):
        super(Decoder_CNN_SBI,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.B = B
        self.iter = Iteration_dec_CNN_SBI(L,dim,B)
        self.demodulation = DeModulator(k,n,L,dim)
        self.dec_out = Conv1D(filters=2,strides=1,kernel_size=1,activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,1)),
                             trainable=True)
    def call(self,x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [tf.shape(x)[0], tf.shape(x)[1], 1])
        de2 = tf.slice(d, [0, 0, 1],[tf.shape(x)[0],tf.shape(x)[1],1])
        dI = tf.slice(d, [0, 0, 2],[tf.shape(x)[0],tf.shape(x)[1],1])
        # shape=(batch_size,L,1)
        p = self.p
        for i in range(5):
            p = self.iter(de1, de2, dI,p)
        # shape=(batch_size,L,1)
        y = self.dec_out(p)
        return y
    def get_config(self):
        config = {
            'batch_size':
                self.batch_size,
            'L':
                self.L,
            'dim':
                self.dim,
            'k':
                self.k,
            'n':
                self.n,
            'B':
                self.B
        }
        base_config = super(Decoder_CNN_SBI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

'''RNN'''
class Decoder_RNN_PRI(tf.keras.layers.Layer):
    def __init__(self, batch_size, L, dim, k, n):
        super(Decoder_RNN_PRI, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.iter = Iteration_dec_RNN_PRI(batch_size,L, dim)
        self.demodulation = DeModulator(k, n,L,dim)
        self.dec_out = Conv1D(filters=2, strides=1, kernel_size=1, activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size, L, 1)),
                             trainable=True)

    def call(self, x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [tf.shape(x)[0], tf.shape(x)[1], 1])
        de2 = tf.slice(d, [0, 0, 1], [tf.shape(x)[0], tf.shape(x)[1], 1])
        dI = tf.slice(d, [0, 0, 2], [tf.shape(x)[0], tf.shape(x)[1], 1])
        # shape=(batch_size,L,1)
        p = self.p
        for i in range(5):
            p = self.iter(de1, de2, dI, p)
        # shape=(batch_size,L,1)
        y = self.dec_out(p)
        return y

    def get_config(self):
        config = {
            'batch_size':
                self.batch_size,
            'L':
                self.L,
            'dim':
                self.dim,
            'k':
                self.k,
            'n':
                self.n
        }
        base_config = super(Decoder_RNN_PRI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Decoder_RNN_SBI(tf.keras.layers.Layer):
    def __init__(self, batch_size, L, dim, k, n, B):
        super(Decoder_RNN_SBI, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.B = B
        self.iter = Iteration_dec_RNN_SBI(L, dim, B)
        self.demodulation = DeModulator(k, n,L,dim)
        self.dec_out = Conv1D(filters=2, strides=1, kernel_size=1, activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size, L, 1)),
                             trainable=True)

    def call(self, x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [tf.shape(x)[0], tf.shape(x)[1], 1])
        de2 = tf.slice(d, [0, 0, 1], [tf.shape(x)[0], tf.shape(x)[1], 1])
        dI = tf.slice(d, [0, 0, 2], [tf.shape(x)[0], tf.shape(x)[1], 1])
        # shape=(batch_size,L,1)
        p = self.p
        for i in range(5):
            p = self.iter(de1, de2, dI, p)
        # shape=(batch_size,L,1)
        y = self.dec_out(p)
        return y

    def get_config(self):
        config = {
            'batch_size':
                self.batch_size,
            'L':
                self.L,
            'dim':
                self.dim,
            'k':
                self.k,
            'n':
                self.n,
            'B':
                self.B
        }
        base_config = super(Decoder_RNN_SBI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
