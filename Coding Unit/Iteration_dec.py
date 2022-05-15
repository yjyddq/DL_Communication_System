import tensorflow as tf
from tensorflow.keras.layers import Conv1D,LSTM,Dense,Bidirectional

from communication.Communication_DL.Coding_Unit.coding_unit import coding_unit
from communication.Communication_DL.Interleaver.Block import Se_Block_Interleaver,Se_Block_DeInterleaver
from communication.Communication_DL.Interleaver.Pseudo_Random import Pseudo_random_Interleaver,Pseudo_random_DeInterleaver

# 迭代单元可以使用两种结构，一种是CNN
# 另一种是BiRNN (LSTM or GRU)
'''CNN'''
class Iteration_dec_CNN_SBI(tf.keras.layers.Layer):
    def __init__(self,L,dim,B):
        super(Iteration_dec_CNN_SBI,self).__init__()
        self.L = L
        self.dim = dim
        self.B = B
        self.decodex1 = coding_unit(dim, 5, 1,'elu')
        self.decodex2 = coding_unit(dim, 5, 1, 'elu')
        self.decodex3 = coding_unit(dim, 5, 1, 'elu')
        self.decodex4 = coding_unit(dim, 5, 1, 'elu')
        self.decodex5 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI1 = coding_unit(dim, 5, 1,'elu')
        self.decodeI2 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI3 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI4 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI5 = coding_unit(dim, 5, 1, 'elu')
        self.SBI = Se_Block_Interleaver(1,B)
        self.DeSBI = Se_Block_DeInterleaver(1,B)
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        q = self.fc1(q)
        # shape=(batch_size, L,1)
        p = self.SBI(q)

        dxI = self.SBI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        q = self.fc2(q)
        # shape=(batch_size, L,1)
        p = self.DeSBI(q)
        # shape=(batch_size,L,1)
        return p

class Iteration_dec_CNN_PRI(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim):
        super(Iteration_dec_CNN_PRI,self).__init__()
        self.L = L
        self.dim = dim
        self.batch_size = batch_size
        self.decodex1 = coding_unit(dim, 5, 1,'elu')
        self.decodex2 = coding_unit(dim, 5, 1, 'elu')
        self.decodex3 = coding_unit(dim, 5, 1, 'elu')
        self.decodex4 = coding_unit(dim, 5, 1, 'elu')
        self.decodex5 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI1 = coding_unit(dim, 5, 1,'elu')
        self.decodeI2 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI3 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI4 = coding_unit(dim, 5, 1, 'elu')
        self.decodeI5 = coding_unit(dim, 5, 1, 'elu')
        self.PRI = Pseudo_random_Interleaver(batch_size,L,1)
        self.DePRI = Pseudo_random_DeInterleaver(batch_size,L,1)
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        q = self.fc1(q)
        # shape=(batch_size, L,1)
        p = self.PRI(q)

        dxI = self.PRI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        q = self.fc2(q)
        # shape=(batch_size, L,1)
        p = self.DePRI(q)
        # shape=(batch_size,L,1)
        return p

'''RNN'''
class Iteration_dec_RNN_SBI(tf.keras.layers.Layer):
    def __init__(self,L,dim,B):
        super(Iteration_dec_RNN_SBI,self).__init__()
        self.L = L
        self.dim = dim
        self.B = B
        self.decodex1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex3 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex4 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex5 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI3 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI4 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI5 = Bidirectional(LSTM(dim,return_sequences=True))
        self.SBI = Se_Block_Interleaver(1,B)
        self.DeSBI = Se_Block_DeInterleaver(1,B)
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        q = self.fc1(q)
        # shape=(batch_size, L,1)
        p = self.SBI(q)

        dxI = self.SBI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        q = self.fc2(q)
        # shape=(batch_size, L,1)
        p = self.DeSBI(q)
        # shape=(batch_size,L,1)
        return p

class Iteration_dec_RNN_PRI(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,dim):
        super(Iteration_dec_RNN_PRI,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.decodex1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex3 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex4 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodex5 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI1 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI2 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI3 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI4 = Bidirectional(LSTM(dim,return_sequences=True))
        self.decodeI5 = Bidirectional(LSTM(dim,return_sequences=True))
        self.PRI = Pseudo_random_Interleaver(batch_size,L,1)
        self.DePRI = Pseudo_random_DeInterleaver(batch_size,L,1)
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        q = self.fc1(q)
        # shape=(batch_size, L,1)
        p = self.PRI(q)

        dxI = self.PRI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        q = self.fc2(q)
        # shape=(batch_size, L,1)
        p = self.DePRI(q)
        # shape=(batch_size,L,1)
        return p