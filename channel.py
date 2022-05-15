import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras import backend as KR
import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Lambda
from communication.Communication_DL.Power_Norm.power_norm import normalization

import numpy as np
# 无训练参数模块都认为是Channel

'''SISO Channel'''
class AWGN_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(AWGN_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,x):
        w = KR.random_normal(KR.shape(x), mean=0.0, stddev=self.noise_sigma)
        y = x + w
        return y

class ISI(tf.keras.layers.Layer):
    def __init__(self,L,k,n,W):
        super(ISI,self).__init__()
        self.W = W # ISI窗口，即t时刻的码元受前W个时刻的码元影响
        self.L = L
        self.k = k
        self.n = n
    def call(self,x):
        y_isi=[]
        xc = real_convert_to_complex(x)
        hc = real_convert_to_complex(KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(1 / 2)))
        isic = tf.multiply(hc, xc)
        isi = complex_convert_to_real(isic)
        for i in range((self.L * self.n)//self.k):
            if i == 0:
                y_isi = tf.zeros(shape=(tf.shape(x)[0],1,tf.shape(x)[-1]))
            elif i < self.W:
                inter = tf.expand_dims(tf.reduce_sum(isi[:, 0:i, :], axis=1),axis=1)
                y_isi = tf.concat([y_isi, inter], axis=1)
            else:
                inter = tf.expand_dims(tf.reduce_sum(isi[:, i-self.W:i, :], axis=1),axis=1)
                y_isi = tf.concat([y_isi,inter],axis=1)
        y_h = y_isi + x
        return y_h

class DeISI(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(DeISI,self).__init__()
        self.diff1 = LSTM(dim,return_sequences=True)
        self.diff2 = LSTM(dim,return_sequences=True)
        self.dense = Dense(2)
    def call(self,x):
        de_isi = self.diff1(x)
        de_isi = self.diff2(de_isi)
        # shape=(batch_size,L*n,dim)
        de_isi = self.dense(de_isi)
        return de_isi


class Rayleigh_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(Rayleigh_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,x):
        # shape=(batch_size,L*n,2)
        xc = real_convert_to_complex(x)
        # shape=(batch_size,L*n,1)
        wc = real_convert_to_complex(KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(self.noise_sigma)))
        hc = real_convert_to_complex(KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(1/2)))
        yc = tf.multiply(hc,xc) + wc
        y = complex_convert_to_real(yc)
        # shape=(batch_size,L*n,2)
        return y

class Bursty_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma,burst_beta,burstyNoise):
        super(Bursty_Channel,self).__init__()
        self.noise_sigma=noise_sigma
        self.burstyNoise = burstyNoise
        self.burst_beta = burst_beta
    def call(self,x):
        w1 = KR.random_normal(KR.shape(x), mean=0.0, stddev=self.noise_sigma)
        w2 = KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(self.burstyNoise))
        y = x + w1 + self.burst_beta*w2
        return y


# 将I-Q 2位实数转换为一位的复数
def real_convert_to_complex(x):
    # shape=(batch_size,L,2)
    x_real = tf.slice(x,[0,0,0],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    x_imag = tf.slice(x,[0,0,tf.shape(x)[-1]//2],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    xc = tf.complex(x_real,x_imag)
    # shape=(batch_size,L,1)
    return xc

# 将I-Q 2位实数转换为一位的复数
def complex_convert_to_real(xc):
    # shape=(batch_size,L,1)
    x_real = tf.math.real(xc)
    x_imag = tf.math.imag(xc)
    x = tf.concat([x_real,x_imag],axis=-1)
    return x


'''MIMO Channel'''
class AWGN_four_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(AWGN_four_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,f):
        # f shape=(batch_size,m,L,n)
        w = KR.random_normal(shape=(tf.shape(f)[0],tf.shape(f)[1],tf.shape(f)[2],2), mean=0.0, stddev=self.noise_sigma)
        wc = real_convert_to_complex(w)
        y = f + wc
        return y

# 正交频分复用
class OFDM(tf.keras.layers.Layer):
    def __init__(self,m):
        super(OFDM,self).__init__()
        self.m = m
        self.power_norm = Lambda(normalization)
    def call(self,x0,x1,x2,x3):
        # shape=(batch_size,L,2)
        x0 = tf.expand_dims(x0, axis=1)
        x1 = tf.expand_dims(x1, axis=1)
        x2 = tf.expand_dims(x2, axis=1)
        x3 = tf.expand_dims(x3, axis=1)
        # shape=(batch_size,1,L,2)
        x = tf.concat([x0,x1,x2,x3],axis=1)
        # shape=(batch_size,m,L,2)
        xc = real_convert_to_complex(x)
        # shape=(batch_size,m,L,1)
        xc = tf.transpose(xc,perm=[0,2,1,3])
        # shape=(batch_size,L,m,1)
        xc = tf.squeeze(xc,axis=-1)
        # fft是只对最后一个维度做fft
        # 所以这里在做fft之前，先要进行维度的压缩
        yc= tf.signal.ifft(xc)*self.m

        yc = tf.expand_dims(yc,axis=-1)
        # shape=(batch_size,L,m,1)
        yc = tf.transpose(yc,perm=[0,2,1,3])
        # shape=(batch_size,m,L,n)
        f = complex_convert_to_real(yc)
        f = self.power_norm(f)
        f = real_convert_to_complex(f)
        return f

# 解复用
class DeOFDM(tf.keras.layers.Layer):
    def __init__(self,m):
        super(DeOFDM,self).__init__()
        self.m = m
    def call(self,x):
        # shape=(batch_size,m,L*n,n)
        x = tf.transpose(x,perm=[0,2,1,3])
        # shape=(batch_size,L*n,m,n)
        x = tf.squeeze(x,axis=-1)
        # shape=(batch_size,L*n,m)
        yc = tf.signal.fft(x)/self.m
        yc = tf.expand_dims(yc,axis=-1)
        yc = tf.transpose(yc,perm=[0,2,1,3])
        # shape=(batch_size,m,L*n,1)
        y = complex_convert_to_real(yc)
        # shape=(batch_size,m,L*n,2)
        x0 = tf.slice(y,[0, 0, 0, 0],  [tf.shape(x)[0],1,tf.shape(x)[2],2])
        x1 = tf.slice(y, [0, 1, 0, 0], [tf.shape(x)[0], 1,tf.shape(x)[2], 2])
        x2 = tf.slice(y, [0, 2, 0, 0], [tf.shape(x)[0], 1,tf.shape(x)[2], 2])
        x3 = tf.slice(y, [0, 3, 0, 0], [tf.shape(x)[0], 1,tf.shape(x)[2], 2])
        x0 = tf.squeeze(x0, axis=1)
        x1 = tf.squeeze(x1, axis=1)
        x2 = tf.squeeze(x2, axis=1)
        x3 = tf.squeeze(x3, axis=1)
        # shape=(batch_size,L*n,2)
        return x0,x1,x2,x3

class Rayleigh_four_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(Rayleigh_four_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,x1,x2,x3,x4):
        # 将实部、虚部转化为复数
        x1c = real_convert_to_complex(x1)
        x2c = real_convert_to_complex(x2)
        x3c = real_convert_to_complex(x3)
        x4c = real_convert_to_complex(x4)
        # 加性噪声
        w1 = real_convert_to_complex(KR.random_normal(KR.shape(x1), mean=0.0, stddev=self.noise_sigma))
        w2 = real_convert_to_complex(KR.random_normal(KR.shape(x2), mean=0.0, stddev=self.noise_sigma))
        w3 = real_convert_to_complex(KR.random_normal(KR.shape(x3), mean=0.0, stddev=self.noise_sigma))
        w4 = real_convert_to_complex(KR.random_normal(KR.shape(x4), mean=0.0, stddev=self.noise_sigma))
        # 乘性噪声
        h_1_1c = real_convert_to_complex(KR.random_normal(KR.shape(x1), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_1_2c = real_convert_to_complex(KR.random_normal(KR.shape(x1), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_1_3c = real_convert_to_complex(KR.random_normal(KR.shape(x1), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_1_4c = real_convert_to_complex(KR.random_normal(KR.shape(x1), mean=0.0, stddev=np.sqrt(1 / 2)))

        h_2_1c = real_convert_to_complex(KR.random_normal(KR.shape(x2), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_2_2c = real_convert_to_complex(KR.random_normal(KR.shape(x2), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_2_3c = real_convert_to_complex(KR.random_normal(KR.shape(x2), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_2_4c = real_convert_to_complex(KR.random_normal(KR.shape(x2), mean=0.0, stddev=np.sqrt(1 / 2)))

        h_3_1c = real_convert_to_complex(KR.random_normal(KR.shape(x3), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_3_2c = real_convert_to_complex(KR.random_normal(KR.shape(x3), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_3_3c = real_convert_to_complex(KR.random_normal(KR.shape(x3), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_3_4c = real_convert_to_complex(KR.random_normal(KR.shape(x3), mean=0.0, stddev=np.sqrt(1 / 2)))

        h_4_1c = real_convert_to_complex(KR.random_normal(KR.shape(x4), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_4_2c = real_convert_to_complex(KR.random_normal(KR.shape(x4), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_4_3c = real_convert_to_complex(KR.random_normal(KR.shape(x4), mean=0.0, stddev=np.sqrt(1 / 2)))
        h_4_4c = real_convert_to_complex(KR.random_normal(KR.shape(x4), mean=0.0, stddev=np.sqrt(1 / 2)))

        y1_complex = tf.multiply(h_1_1c,x1c)+tf.multiply(h_2_1c,x2c)+\
             tf.multiply(h_3_1c,x3c)+tf.multiply(h_4_1c,x4c) + w1

        y2_complex = tf.multiply(h_1_2c, x1c) + tf.multiply(h_2_2c, x2c) + \
             tf.multiply(h_3_2c, x3c) + tf.multiply(h_4_2c, x4c) + w2

        y3_complex = tf.multiply(h_1_3c, x1c) + tf.multiply(h_2_3c, x2c) + \
             tf.multiply(h_3_3c, x3c) + tf.multiply(h_4_3c, x4c) + w3

        y4_complex = tf.multiply(h_1_4c, x1c) + tf.multiply(h_2_4c, x2c) + \
             tf.multiply(h_3_4c, x3c) + tf.multiply(h_4_4c, x4c) + w4
        # 提取复数的实部虚部
        y1_real = tf.math.real(y1_complex)
        y1_imag = tf.math.imag(y1_complex)
        y1 = tf.concat([y1_real,y1_imag],axis=-1)

        y2_real = tf.math.real(y2_complex)
        y2_imag = tf.math.imag(y2_complex)
        y2 = tf.concat([y2_real, y2_imag], axis=-1)

        y3_real = tf.math.real(y3_complex)
        y3_imag = tf.math.imag(y3_complex)
        y3 = tf.concat([y3_real, y3_imag], axis=-1)

        y4_real = tf.math.real(y4_complex)
        y4_imag = tf.math.imag(y4_complex)
        y4 = tf.concat([y4_real, y4_imag], axis=-1)
        # shape=(batch_size,L*n,2)
        return y1,y2,y3,y4

def Construct_Hermit(num,batch_size,Ln=None):
    # num从2起,取2**n
    Z = []
    a_base = tf.cast([[1,1],[0,0]],tf.float32)
    b_base = tf.cast([[0,0],[1,-1]],tf.float32)
    z = tf.complex(a_base, b_base) # shape=(2,2)
    for i in range(round(np.log2(num))):
        if i == 0:
            Z = z
        elif i == 1:
            Z_up = tf.concat([z,z],axis=1) # shape=(2,4)
            Z_down = tf.concat([z,-z],axis=1) # shape=(2,4)
            Z = tf.concat([Z_up,Z_down],axis=0) # shape=(4,4)
        else:
            Z_up = tf.concat([Z, Z], axis=1)  # shape=(2,4)
            Z_down = tf.concat([Z, -Z], axis=1)  # shape=(2,4)
            Z = tf.concat([Z_up, Z_down], axis=0)  # shape=(4,4)
    # Z = Z/np.sqrt(num)
    Zconj = tf.math.conj(Z) # 共轭
    ZjT = tf.transpose(Zconj) # 转置
    # shape=(num,num)
    Z = tf.expand_dims(Z, axis=0)
    ZjT = tf.expand_dims(ZjT, axis=0)
    Z = tf.broadcast_to(Z, [batch_size, num, num])
    ZjT = tf.broadcast_to(ZjT, [batch_size, num, num])
    # 均为复数
    if Ln != None:
        Z = tf.expand_dims(Z, axis=1)
        ZjT = tf.expand_dims(ZjT, axis=1)
        Z = tf.broadcast_to(Z, [batch_size,Ln, num, num])
        ZjT = tf.broadcast_to(ZjT, [batch_size,Ln, num, num])
    return Z,ZjT

class Code_Multiplexing(tf.keras.layers.Layer):
    def __init__(self,L,n,num,batch_size,trainable=False):
        super(Code_Multiplexing,self).__init__()
        self.L = L
        self.n = n
        self.num = num
        self.batch_size = batch_size
        self.Z, _ = tf.Variable(Construct_Hermit(num,batch_size,L*n),trainable=trainable)
    def call(self,x0,x1,x2,x3):
        # x shape=(batch_size,L*n,2)
        x0c = tf.expand_dims(real_convert_to_complex(x0),axis=1)
        x1c = tf.expand_dims(real_convert_to_complex(x1), axis=1)
        x2c = tf.expand_dims(real_convert_to_complex(x2), axis=1)
        x3c = tf.expand_dims(real_convert_to_complex(x3), axis=1)
        # x shape=(batch_size,1,L*n,1)
        xc = tf.concat([x0c,x1c,x2c,x3c],axis=1)
        # x shape=(batch_size,m,L*n,1)
        # hermit shape=(batch_size,L*n,num,num)
        # 将L*n长度的数据按照num进行分组
        xc = tf.transpose(xc,perm=[0,2,1,3])
        # x shape=(batch_size,L*n,m,1)
        yc = tf.matmul(self.Z,xc)
        y0c = tf.squeeze(tf.slice(yc,[0,0,0,0],[self.batch_size,1,self.L*self.n,1]),axis=1)
        y1c = tf.squeeze(tf.slice(yc, [0, 1, 0, 0], [self.batch_size, 1, self.L * self.n, 1]),axis=1)
        y2c = tf.squeeze(tf.slice(yc, [0, 2, 0, 0], [self.batch_size, 1, self.L * self.n, 1]),axis=1)
        y3c = tf.squeeze(tf.slice(yc, [0, 3, 0, 0], [self.batch_size, 1, self.L * self.n, 1]),axis=1)
        y0 = complex_convert_to_real(y0c)
        y1 = complex_convert_to_real(y1c)
        y2 = complex_convert_to_real(y2c)
        y3 = complex_convert_to_real(y3c)
        # y shape=(batch_size,L*n,2)
        return y0,y1,y2,y3

class Code_DeMultiplexing(tf.keras.layers.Layer):
    def __init__(self,L,n,num,batch_size,trainable=False):
        super(Code_DeMultiplexing,self).__init__()
        self.L = L
        self.n = n
        self.num = num
        self.batch_size = batch_size
        _, self.ZjT = tf.Variable(Construct_Hermit(num,batch_size,L*n),trainable=trainable)
    def call(self,x0,x1,x2,x3):
        # shape=(batch_size,L*n,num,num)
        x0c = tf.expand_dims(real_convert_to_complex(x0), axis=1)
        x1c = tf.expand_dims(real_convert_to_complex(x1), axis=1)
        x2c = tf.expand_dims(real_convert_to_complex(x2), axis=1)
        x3c = tf.expand_dims(real_convert_to_complex(x3), axis=1)
        xc = tf.concat([x0c, x1c, x2c, x3c], axis=1)

        xc = tf.transpose(xc, perm=[0, 2, 1, 3])
        # x shape=(batch_size,L*n,m,1)
        yc = tf.matmul(self.ZjT, xc)

        y0c = tf.squeeze(tf.slice(yc, [0, 0, 0, 0], [self.batch_size, 1, self.L * self.n, 1]), axis=1)
        y1c = tf.squeeze(tf.slice(yc, [0, 1, 0, 0], [self.batch_size, 1, self.L * self.n, 1]), axis=1)
        y2c = tf.squeeze(tf.slice(yc, [0, 2, 0, 0], [self.batch_size, 1, self.L * self.n, 1]), axis=1)
        y3c = tf.squeeze(tf.slice(yc, [0, 3, 0, 0], [self.batch_size, 1, self.L * self.n, 1]), axis=1)
        y0 = complex_convert_to_real(y0c)
        y1 = complex_convert_to_real(y1c)
        y2 = complex_convert_to_real(y2c)
        y3 = complex_convert_to_real(y3c)
        # y shape=(batch_size,L*n,2)
        return y0,y1,y2,y3

