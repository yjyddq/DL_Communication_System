import tensorflow as tf
from tensorflow.keras.layers import LSTM,Dense,Bidirectional

from DL_Communication_System.Coding_Unit.coding_unit import coding_unit
from DL_Communication_System.Interleaver.Block import Se_Block_Interleaver
from DL_Communication_system.Interleaver.Pseudo_Random import Pseudo_random_Interleaver
from DL_Communication_System.Coding_Unit.Modulation import Modulator

'''CNN'''
class Encoder_CNN_PRI(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim,k,n):
        super(Encoder_CNN_PRI,self).__init__()
        self.k = k
        self.n = n
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.codex1 = coding_unit(dim,5,1,'elu')
        self.codex2 = coding_unit(dim,5,1,'elu')
        self.codeI = coding_unit(dim,5,1,'elu')
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.PRI = Pseudo_random_Interleaver(batch_size,L,1)
        self.modulation = Modulator(k,n,L,dim)
    def call(self,x):
        x = 2 * x - 1
        # shape=(batch_size,L,1)
        e1 = self.codex1(x)
        e1 = self.fc1(e1)

        e2 = self.codex2(x)
        e2 = self.fc2(e2)

        xI = self.PRI(x)
        eI = self.codeI(xI)
        eI = self.fcI(eI)

        y = tf.concat([e1,e2,eI],axis=-1)
        # shape=(batch_size,L,n)
        y = self.modulation(y)
        # # shape = (batch_size, L * n//k, 2)
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
        base_config = super(Encoder_CNN_PRI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Encoder_CNN_SBI(tf.keras.layers.Layer):
    def __init__(self, B, L, dim,k,n):
        super(Encoder_CNN_SBI, self).__init__()
        self.k = k
        self.n = n
        self.B = B
        self.L = L
        self.dim = dim
        self.codex1 = coding_unit(dim, 5, 1, 'elu')
        self.codex2 = coding_unit(dim, 5, 1, 'elu')
        self.codeI = coding_unit(dim, 5, 1, 'elu')
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.SBI = Se_Block_Interleaver(1, B)
        self.modulation = Modulator(k,n,L,dim)

    def call(self, x):
        x = 2 * x - 1
        # shape=(batch_size,L,1)
        e1 = self.codex1(x)
        e1 = self.fc1(e1)

        e2 = self.codex2(x)
        e2 = self.fc2(e2)

        xI = self.SBI(x)
        eI = self.codeI(xI)
        eI = self.fcI(eI)

        y = tf.concat([e1, e2, eI], axis=-1)
        # shape=(batch_size,L,n)
        y = self.modulation(y)
        # # shape = (batch_size, L * n//k, 2)
        return y

    def get_config(self):
        config = {
            'k':
                self.k,
            'n':
                self.n,
            'B':
                self.B,
            'L':
                self.L,
            'dim':
                self.dim
        }
        base_config = super(Encoder_CNN_SBI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


'''RNN'''
class Encoder_RNN_PRI(tf.keras.layers.Layer):
    def __init__(self, batch_size, L, dim,k,n):
        super(Encoder_RNN_PRI, self).__init__()
        self.k = k
        self.n = n
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.codex1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.codex2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.codeI = Bidirectional(LSTM(dim,return_sequences=True))
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.PRI = Pseudo_random_Interleaver(batch_size,L,1)
        self.modulation = Modulator(k,n,L,dim)

    def call(self, x):
        x = 2 * x - 1
        # shape=(batch_size,L,1)
        e1 = self.codex1(x)
        e1 = self.fc1(e1)

        e2 = self.codex2(x)
        e2 = self.fc2(e2)

        xI = self.PRI(x)
        eI = self.codeI(xI)
        eI = self.fcI(eI)

        y = tf.concat([e1, e2, eI], axis=-1)
        # shape=(batch_size,L,n)
        y = self.modulation(y)
        # # shape = (batch_size, L * n//k, 2)
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
        base_config = super(Encoder_RNN_PRI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Encoder_RNN_SBI(tf.keras.layers.Layer):
    def __init__(self, B, L, dim,k,n):
        super(Encoder_RNN_SBI, self).__init__()
        self.k = k
        self.n = n
        self.B = B
        self.L = L
        self.dim = dim
        self.codex1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.codex2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.codeI = Bidirectional(LSTM(dim,return_sequences=True))
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.SBI = Se_Block_Interleaver(1, B)
        self.modulation = Modulator(k,n,L,dim)

    def call(self, x):
        x = 2 * x - 1
        # shape=(batch_size,L,1)
        e1 = self.codex1(x)
        e1 = self.fc1(e1)

        e2 = self.codex2(x)
        e2 = self.fc2(e2)

        xI = self.PRI(x)
        eI = self.codeI(xI)
        eI = self.fcI(eI)

        y = tf.concat([e1, e2, eI], axis=-1)
        # shape=(batch_size,L,n)
        y = self.modulation(y)
        # # shape = (batch_size, L * n//k, 2)
        return y

    def get_config(self):
        config = {
            'k':
                self.k,
            'n':
                self.n,
            'B':
                self.B,
            'L':
                self.L,
            'dim':
                self.dim
        }
        base_config = super(Encoder_RNN_SBI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
