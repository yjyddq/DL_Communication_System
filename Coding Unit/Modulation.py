import tensorflow as tf
from communication.Communication_DL.Coding_Unit.coding_unit import coding_unit


class Modulator(tf.keras.layers.Layer):
    def __init__(self,k,n,L,dim):
        super(Modulator,self).__init__()
        self.L = L
        self.n = n
        self.k = k
        self.dim = dim
        self.map = coding_unit(dim,1,1,'elu')
        self.modulation = coding_unit(2,1,1,'linear')
    def call(self,x):
        # x shape=(batch_size,L,n)
        xtmp = tf.reshape(x,shape=(tf.shape(x)[0],(self.L*self.n)//self.k,self.k))
        # shape = (batch_size, L * n//k, k)
        mo = self.map(xtmp)
        # shape = (batch_size, L * n//k, dim)
        coordinate = self.modulation(mo)
        # shape = (batch_size, L * n//k, 2)
        return coordinate


class DeModulator(tf.keras.layers.Layer):
    def __init__(self,k,n,L,dim):
        super(DeModulator,self).__init__()
        self.k = k
        self.L = L
        self.n = n
        self.dim = dim
        self.remap = coding_unit(dim,1,1,'elu')
        self.demodulation = coding_unit(k,1,1,'linear')
    def call(self,x):
        # x shape = (batch_size, L * n//k, 2)
        rm = self.remap(x)
        # shape = (batch_size, L * n//k, dim)
        codeword = self.demodulation(rm)
        # shape = (batch_size, L * n//k, k)
        bitseq = tf.reshape(codeword,shape=(tf.shape(x)[0],self.L,self.n))
        # shape = (batch_size, L , n)
        return bitseq
