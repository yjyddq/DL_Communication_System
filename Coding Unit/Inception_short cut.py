import tensorflow as tf
from tensorflow.keras.layers import Conv1D,Dense,Activation

from communication.Communication_DL.Coding_Unit.coding_unit import coding_unit
from communication.Communication_DL.Interleaver.Pseudo_Random import Pseudo_random_Interleaver,Pseudo_random_DeInterleaver


class Inception_Block(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(Inception_Block,self).__init__()
        self.conv3 = coding_unit(dim,3,1,'elu')
        self.conv5 = coding_unit(dim,5,1,'elu')
        self.conv7 = coding_unit(dim,7, 1, 'elu')
        self.conv3_1 = coding_unit(dim // 5,1,1,'elu')
        self.conv5_1 = coding_unit(dim // 5, 1, 1, 'elu')
        self.conv7_1 = coding_unit(dim // 5, 1, 1, 'elu')
    def call(self,x):
        x3_1 = self.conv3_1(x)
        x5_1 = self.conv5_1(x)
        x7_1 = self.conv7_1(x)
        x3 = self.conv3(x3_1)
        x5 = self.conv5(x5_1)
        x7 = self.conv7(x7_1)
        y = tf.concat([x3,x5,x7],axis=-1)
        return y

class Modulator(tf.keras.layers.Layer):
    def __init__(self,k,n,L,dim):
        super(Modulator,self).__init__()
        self.L = L
        self.n = n
        self.k = k
        self.dim = dim
        # 调制是将bit打包，然后映射为星座图上的坐标点
        self.map = coding_unit(dim,1,1,'elu')
        self.modulation = coding_unit(2,1,1,'linear')
    def call(self,x):
        # x shape=(batch_size,L,n*dim)
        xtmp = tf.reshape(x,shape=(tf.shape(x)[0],(self.L*self.n)//self.k,self.k*self.dim))
        # shape = (batch_size, L * n//k, k*dim)
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
        self.demodulation = coding_unit(k*dim,1,1,'linear')
    def call(self,x):
        # x shape = (batch_size, L * n//k, 2)
        rm = self.remap(x)
        # shape = (batch_size, L * n//k, dim)
        codeword = self.demodulation(rm)
        # shape = (batch_size, L * n//k, k*dim)
        bitseq = tf.reshape(codeword,shape=(tf.shape(x)[0],self.L,self.n*self.dim))
        # shape = (batch_size, L , n)
        return bitseq

class Encoder(tf.keras.layers.Layer):
    def __init__(self,batch_size, L,dim,k,n):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.codex1_1 = Inception_Block(dim)
        self.codex1_2 = Inception_Block(dim)
        self.codex2_1 = Inception_Block(dim)
        self.codex2_2 = Inception_Block(dim)
        self.codeI_1 = Inception_Block(dim)
        self.codeI_2 = Inception_Block(dim)
        self.identical_map1 = Dense(dim)
        self.identical_map2 = Dense(dim)
        self.identical_mapI = Dense(dim)
        self.PRI = Pseudo_random_Interleaver(batch_size,L, 2)
        self.residual_act = Activation('elu')
        self.modulation = Modulator(k,n,L,dim)

    def call(self, x):
        # shape=(batch_size,L,2)

        e1 = self.codex1_1(x)
        e1 = self.codex1_2(e1)
        x1 = self.identical_map1(x)
        e1 = self.residual_act(e1 + x1)

        e2 = self.codex2_1(x)
        e2 = self.codex2_2(e2)
        x2 = self.identical_map2(x)
        e2 = self.residual_act(e2 + x2)

        xI = self.PRI(x)
        eI = self.codeI_1(xI)
        eI = self.codeI_2(eI)
        xI = self.identical_mapI(xI)
        eI = self.residual_act(eI + xI)

        y = tf.concat([e1, e2, eI], axis=-1)
        # shape=(batch_size,L,n*dim)
        y = self.modulation(y)
        # y = self.power_norm(y)
        # # shape = (batch_size, L * n//m, 2)
        return y

    def get_config(self):
        config = {
            'k':
                self.k,
            'n':
                self.n,
            'batch_size':
                self.batch_size,
            'L':
                self.L,
            'dim':
                self.dim
        }
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Iteration_dec(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim):
        super(Iteration_dec,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.decodex1 = Inception_Block(dim)
        self.decodex2 = Inception_Block(dim)
        self.decodex3 = Inception_Block(dim)
        self.decodex4 = Inception_Block(dim)
        self.decodex5 = Inception_Block(dim)
        self.decodeI1 = Inception_Block(dim)
        self.decodeI2 = Inception_Block(dim)
        self.decodeI3 = Inception_Block(dim)
        self.decodeI4 = Inception_Block(dim)
        self.decodeI5 = Inception_Block(dim)
        self.Identical_map = Dense(dim)
        self.Identical_mapI = Dense(dim)
        self.residual_act=Activation('elu')
        self.PRI = Pseudo_random_Interleaver(batch_size,L,dim)
        self.DePRI = Pseudo_random_DeInterleaver(batch_size,L,dim)

    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        tmp_Identical = self.Identical_map(tmp)
        q = self.residual_act(q+tmp_Identical)
        # shape=(batch_size, L,1)
        p = self.PRI(q)

        dxI = self.PRI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        tmp_Identical_I = self.Identical_mapI(tmp)
        q = self.residual_act(q+tmp_Identical_I)
        # shape=(batch_size, L,1)
        p = self.DePRI(q)
        # shape=(batch_size,L,1)
        return p

class Decoder(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim,k,n):
        super(Decoder,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.k = k
        self.n = n
        self.iter = Iteration_dec(batch_size,L,dim)
        self.demodulation = DeModulator(k,n,L,dim)
        self.dec_out = Conv1D(filters=2,strides=1,kernel_size=1,activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,dim)),
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
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))