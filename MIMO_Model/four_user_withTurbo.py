# 使用OFDM调制方案
# AWGN信道下，使用OFDM足够能解决问题
# 但是Rayleigh信道时，OFDM欠佳
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, Flatten, Activation,Conv1DTranspose,Embedding,LayerNormalization,LocallyConnected1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as KR
import numpy as np
import copy
import time
import tensorflow as tf

import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

# from communication.Turbo_DL.Coding_Unit.coding_unit import coding_unit
# from communication.Turbo_DL.Interleaver.Block import Se_Block_Interleaver,Se_Block_DeInterleaver
# from communication.Turbo_DL.Interleaver.Pseudo_random import Pseudo_random_Interleaver,Pseudo_random_DeInterleaver
# from communication.Turbo_DL.Modulation.Modulation import Modulation
# from communication.Turbo_DL.Modulation.DeModulation import DeModulation
# from communication.Turbo_DL.Coding_Unit.Iteration_dec import Iteration_dec_CNN_PI
# from communication.Turbo_DL.Channel.Channel import normalization,ISI,DeISI,AWGN_Channel,Bursty_Channel,Multiplexing,DeMultiplexing,Rayleigh_Channel

'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 1

# Number of symbols
L = 100

# Channel Use
n = 3

dim = 50
# User number
m = 4
# Effective Throughput
#  (bits per symbol)*( number of symbols) / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 2

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 64
nb_train_word = batch_size*200

'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
# User one
train_data0 = np.random.randint(low=0, high=2, size=(nb_train_word, L))
train_data0 = np.reshape(train_data0, newshape=(nb_train_word, L, 1))


# Convert Integer Data to one-hot vector
vec_one_hot0 = to_categorical(y=train_data0, num_classes=2)

# User two
train_data1 = np.random.randint(low=0, high=2, size=(nb_train_word, L))
train_data1 = np.reshape(train_data1, newshape=(nb_train_word, L, 1))
vec_one_hot1 = to_categorical(y=train_data1, num_classes=2)

# User three
train_data2 = np.random.randint(low=0, high=2, size=(nb_train_word, L))
train_data2 = np.reshape(train_data2, newshape=(nb_train_word, L, 1))
vec_one_hot2 = to_categorical(y=train_data2, num_classes=2)

# User four
train_data3 = np.random.randint(low=0, high=2, size=(nb_train_word, L))
train_data3 = np.reshape(train_data3, newshape=(nb_train_word, L, 1))
vec_one_hot3 = to_categorical(y=train_data3, num_classes=2)

# tensorflow的离散傅里叶变换不除以N
# 在做离散傅里叶逆变换的时候除以N
# tf.signal.fft()
# tf.signal.ifft()
# 将两个实数转为一个复数
# tf.complex(real,image)
# 在MIMO系统时可以使用OFDM调制
# OFDM调制用FFT实现

'''
 --- NEURAL NETWORKS PARAMETERS ---
'''

early_stopping_patience = 100

epochs = 100

optimizer = Adam(learning_rate=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)


# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)

# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'model_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)

'''
 --- CUSTOMIZED NETWORK LAYERS ---
'''



# 将I-Q 2位实数转换为一位的复数
def real_convert_to_complex(x):
    # shape=(batch_size,m,L,2)
    xc = tf.expand_dims(tf.complex(x[:,:,:,0],x[:,:,:,1]),
                           axis=-1)
    # shape=(batch_size,m,L,1))
    return xc

# 将I-Q 2位实数转换为一位的复数
def complex_convert_to_real(xc):
    # shape=(batch_size,m,L,1)
    x_real = tf.math.real(xc)
    x_imag = tf.math.imag(xc)
    x = tf.concat([x_real,x_imag],axis=-1)
    return x

def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels

class coding_unit(tf.keras.layers.Layer):
    def __init__(self,dim,kernel_size,strides,act):
        super(coding_unit,self).__init__()
        self.dim=dim
        self.conv=Conv1D(filters=dim, strides=strides, kernel_size=kernel_size,padding='same')
        self.BN=BatchNormalization()
        self.act=Activation(act)
    def call(self,x):
        y=self.conv(x)
        y=self.BN(y)
        y=self.act(y)
        return y

class Pseudo_random_Interleaver(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Pseudo_random_Interleaver,self).__init__()
        self.L = L # 一帧的长度
    def call(self,x):
        # shape=(batch_size,L,1)
        p = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(batch_size):
            xtmp = tf.slice(x,[i,0,0],[1,self.L,1])
            # shape=(1,L,1)
            xtmp = tf.reshape(xtmp,shape=(self.L,1))
            np.random.seed(i)
            mshuf = np.random.permutation(mseq)
            ptmp = tf.gather(xtmp,mshuf)
            # shape=(L)
            ptmp = tf.reshape(ptmp,shape=(1,self.L,1))
            # shape=(1,L,1)
            if i == 0:
                p = ptmp
            else:
                p = tf.concat([p,ptmp],axis=0)
                # shape=(batch_size,L,1)
        return p

class Pseudo_random_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Pseudo_random_DeInterleaver,self).__init__()
        self.L = L # 一帧的长度
    def call(self,x):
        # shape=(batch_size,L,1)
        y = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(batch_size):
            xtmp = tf.slice(x,[i,0,0],[1,self.L,1])
            # shape=(1,L,1)
            xtmp = tf.reshape(xtmp, shape=(self.L,1))
            # shape=(L)
            np.random.seed(i)
            # 将标号按照相同的seed进行打乱
            mshuf = np.random.permutation(mseq)
            # argsort默认按照从小到大排序，返回值为对应元素索引值
            # 即得到恢复mshuf正常顺序的index，同理这也是被打乱bit的恢复index
            indices = tf.argsort(mshuf)
            ytmp = tf.gather(xtmp,indices)
            ytmp = tf.reshape(ytmp,shape=(1,self.L,1))
            # shape=(1,L,1)
            if i == 0:
                y = ytmp
            else:
                y = tf.concat([y,ytmp],axis=0)
                # shape=(batch_size,L,1)
        return y

class Modulation(tf.keras.layers.Layer):
    def __init__(self,k):
        super(Modulation,self).__init__()
        self.k = k
        # 调制是将bit打包，然后映射为星座图上的坐标点
        self.map = coding_unit(dim,1,1,'elu')
        self.modulation = coding_unit(2,1,1,'linear')
    def call(self,x):
        # x shape=(batch_size,L,n)
        xtmp = tf.reshape(x, shape=(batch_size, (L * n) // k, k))
        # k个bit一起打包
        mo = self.map(xtmp)
        # shape = (batch_size, L*n//k ,dim)
        coordinate = self.modulation(mo)
        # shape = (batch_size, L*n//k ,2)
        # 分成k个时隙进行发送
        return coordinate

class EncoderSISO(tf.keras.layers.Layer):
    def __init__(self,L):
        super(EncoderSISO,self).__init__()
        self.L = L
        self.codex1 = coding_unit(dim,5,1,'elu')
        self.codex2 = coding_unit(dim,5, 1, 'elu')
        self.codeI = coding_unit(dim,5,1,'elu')
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.PRI = Pseudo_random_Interleaver(L)
        self.modulation = Modulation(k)
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
        # x_BPSK = 2*x-1
        # y = tf.concat([x_BPSK,e,eI],axis=-1)
        y = tf.concat([e1,e2,eI],axis=-1)
        # shape=(batch_size,L,n)
        y = self.modulation(y)
        # y = self.power_norm(y)
        # # shape = (batch_size, L * n//k, 2)
        return y
    def get_config(self):
        config = {
            'L':
                self.L
        }
        base_config = super(EncoderSISO, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        yc= tf.signal.ifft(xc)*m

        yc = tf.expand_dims(yc,axis=-1)
        # shape=(batch_size,L,m,1)
        yc = tf.transpose(yc,perm=[0,2,1,3])
        # shape=(batch_size,m,L,n)
        f = complex_convert_to_real(yc)
        f = self.power_norm(f)
        f = real_convert_to_complex(f)
        return f

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

class Iteration_dec(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Iteration_dec,self).__init__()
        self.L = L
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
        self.PRI = Pseudo_random_Interleaver(L)
        self.DePRI = Pseudo_random_DeInterleaver(L)
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

class DeModulation(tf.keras.layers.Layer):
    def __init__(self,n):
        super(DeModulation,self).__init__()
        self.n = n
        # 调制是将bit打包，然后映射为星座图上的坐标点
        self.remap = coding_unit(dim,1,1,'elu')
        self.demodulation = coding_unit(k,1,1,'linear')
    def call(self,x):
        # x shape = (batch_size, (L*n)//k, 2)
        rm = self.remap(x)
        # shape = (batch_size, (L*n)//k , dim)
        codeword = self.demodulation(rm)
        # shape = (batch_size, (L*n)//k, k)
        bitseq = tf.reshape(codeword, shape=(batch_size, L, n))
        return bitseq

class DecoderSISO(tf.keras.layers.Layer):
    def __init__(self,L):
        super(DecoderSISO,self).__init__()
        self.L = L
        self.iter = Iteration_dec(L)
        self.demodulation = DeModulation(n)
        self.dec_out = Conv1D(filters=2,strides=1,kernel_size=1,activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,1)),
                             trainable=True)
        self.seed = 0
    def call(self,x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [batch_size, L, 1])
        de2 = tf.slice(d, [0, 0, 1],[batch_size,L,1])
        dI = tf.slice(d, [0, 0, 2],[batch_size,L,1])
        # shape=(batch_size,L,1)
        p = self.p
        for i in range(5):
            p = self.iter(de1, de2, dI,p)
        # shape=(batch_size,L,1)
        y = self.dec_out(p)
        return y
    def get_config(self):
        config = {
            'L':
                self.L
        }
        base_config = super(DecoderSISO, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        yc = tf.signal.fft(x)/m
        yc = tf.expand_dims(yc,axis=-1)
        yc = tf.transpose(yc,perm=[0,2,1,3])
        # shape=(batch_size,m,L*n,n)
        y = complex_convert_to_real(yc)
        # shape=(batch_size,m,L*n,2)
        x0 = tf.slice(y,[0, 0, 0, 0],  [batch_size,1,L*n,2])
        x1 = tf.slice(y, [0, 1, 0, 0], [batch_size, 1, L*n, 2])
        x2 = tf.slice(y, [0, 2, 0, 0], [batch_size, 1, L*n, 2])
        x3 = tf.slice(y, [0, 3, 0, 0], [batch_size, 1, L*n, 2])
        x0 = tf.squeeze(x0, axis=1)
        x1 = tf.squeeze(x1, axis=1)
        x2 = tf.squeeze(x2, axis=1)
        x3 = tf.squeeze(x3, axis=1)
        # shape=(batch_size,L*n,2)
        return x0,x1,x2,x3


model_input0 = Input(batch_shape=(batch_size, L,1))
model_input1 = Input(batch_shape=(batch_size, L,1))
model_input2 = Input(batch_shape=(batch_size, L,1))
model_input3 = Input(batch_shape=(batch_size, L,1))

# e = EncoderMIMO(m,dim,n)(model_input0,model_input1,model_input2,model_input3)
e0 = EncoderSISO(L)(model_input0)
e1 = EncoderSISO(L)(model_input1)
e2 = EncoderSISO(L)(model_input2)
e3 = EncoderSISO(L)(model_input3)

f = OFDM(m)(e0,e1,e2,e3)

y_h = AWGN_four_Channel(noise_sigma)(f)

# Output One hot vector and use Softmax to soft decoding
# model_output0,model_output1,model_output2,model_output3 = DecoderMIMO(m,dim,k)(y_h)
d0,d1,d2,d3=DeOFDM(m)(y_h)

model_output0 = DecoderSISO(L)(d0)
model_output1 = DecoderSISO(L)(d1)
model_output2 = DecoderSISO(L)(d2)
model_output3 = DecoderSISO(L)(d3)
# Build System Model
sys_model = Model([model_input0,model_input1,model_input2,model_input3], [model_output0,model_output1,model_output2,model_output3])
# print(sys_model.layers[4].input)
# print(sys_model.layers[4].output)
# sys_model.get_layer()
# 用于观察星座图
# encoder0 = Model(model_input0,e0)
# encoder1 = Model(model_input1,e1)
# encoder2 = Model(model_input2,e2)
# encoder3 = Model(model_input3,e3)
# decoder0 = Model(d0,model_output0)
# decoder1 = Model(d1,model_output1)
# decoder2 = Model(d2,model_output2)
# decoder3 = Model(d3,model_output3)
# 编码器和解码器都要拆开加载权重参数
# 训练的时候记得将解码器的权重参数进行保存
# Functional 对象不能通过这种方式加载权重参数
# path_encoder0='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder0.load_weights(path_encoder0)
# path_encoder1='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder1.load_weights(path_encoder1)
# path_encoder2='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder2.load_weights(path_encoder2)
# path_encoder3='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder3.load_weights(path_encoder3)
#
# path_decoder0='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_decoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder0.load_weights(path_decoder0)
# path_decoder1='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_decoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder1.load_weights(path_decoder1)
# path_decoder2='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_decoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder2.load_weights(path_decoder2)
# path_decoder3='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_decoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder3.load_weights(path_decoder3)
# Print Model Architecture
sys_model.summary()
encoder=sys_model.layers[4]
# encoder = Model(model_input0,e0)

class AWGN_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(AWGN_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,x):
        w = KR.random_normal(KR.shape(x), mean=0.0, stddev=self.noise_sigma)
        y = x + w
        return y

model_input = Input(batch_shape=(batch_size, L,1), name='input_bits')

e = EncoderSISO(L)(model_input)

e_power = Lambda(normalization)(e)

# e_isi = ISI(W)(e_power)
# e_power = Lambda(Norm)(e)

y_h = AWGN_Channel(noise_sigma)(e_power)

# de_isi = DeISI(dim)(y_h)

model_output=DecoderSISO(L)(y_h)
# Build System Model
Turbo_model = Model(model_input, model_output)
Turbo_path = 'E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
Turbo_model.load_weights(Turbo_path)
# path_encoder='E:/pycharm/communication/Memory_code/paper_iter' + '/DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(2) + 'dB' + ' ' + 'AWGN' + '.h5'
encoder.set_weights(Turbo_model.layers[1].get_weights())
print('success')
print(sys_model.layers[-1])
print(sys_model.layers[-2])
print(sys_model.layers[-3])
print(sys_model.layers[-4])
# def custom_loss(real,pred):
#     L1 = tf.reduce_mean(-tf.reduce_sum(real[0] * tf.math.log(pred[0])))
#     L2 = tf.reduce_mean(-tf.reduce_sum(real[1] * tf.math.log(pred[1])))
#     L3 = tf.reduce_mean(-tf.reduce_sum(real[2] * tf.math.log(pred[2])))
#     L4 = tf.reduce_mean(-tf.reduce_sum(real[3] * tf.math.log(pred[3])))
#     alpha1 = L1 / (L1 + L2 + L3 + L4)
#     alpha2 = L2 / (L1 + L2 + L3 + L4)
#     alpha3 = L3 / (L1 + L2 + L3 + L4)
#     alpha4 = L4 / (L1 + L2 + L3 + L4)
#     L = alpha1 * L1 + alpha2 * L2 + alpha3 * L3 + alpha4 * L4
#     return L
# # Compile Model
# sys_model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])
# # print('encoder output:', '\n', encoder.predict(vec_one_hot, batch_size=batch_size))
#
# print('starting train the NN...')
# start = time.perf_counter()
#
#
# # TRAINING
# with tf.device("/cpu:0"):
#     mod_history = sys_model.fit([train_data0,train_data1,train_data2,train_data3], [vec_one_hot0,vec_one_hot1,vec_one_hot2,vec_one_hot3],
#                                 batch_size=batch_size,
#                                 epochs=epochs,
#                                 verbose=1,
#                                 shuffle=True,
#                                 validation_split=0.3, callbacks=[modelcheckpoint])
#
# encoder0_path='./' + 'encoder0_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder0.save(encoder0_path)
#
# encoder1_path='./' + 'encoder1_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder1.save(encoder1_path)
#
# encoder2_path='./' + 'encoder2_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder2.save(encoder2_path)
#
# encoder3_path='./' + 'encoder3_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder3.save(encoder3_path)
#
# decoder0_path='./' + 'decoder0_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder0.save(decoder0_path)
#
# decoder1_path='./' + 'decoder1_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder1.save(decoder1_path)
#
# decoder2_path='./' + 'decoder2_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder2.save(decoder2_path)
#
# decoder3_path='./' + 'decoder3_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder3.save(decoder3_path)
#
# end = time.perf_counter()
#
# print('The NN has trained ' + str(end - start) + ' s')
#
#
# # Plot the Training Loss and Validation Loss
# hist_dict = mod_history.history
#
# # val_loss = hist_dict['val_loss']
# loss = hist_dict['loss']
# decoder0_loss= hist_dict['decoder_siso_loss']
# decoder1_loss= hist_dict['decoder_siso_1_loss']
# decoder2_loss= hist_dict['decoder_siso_2_loss']
# decoder3_loss= hist_dict['decoder_siso_3_loss']
# decoder0_accuracy = hist_dict['decoder_siso_accuracy']
# decoder1_accuracy = hist_dict['decoder_siso_1_accuracy']
# decoder2_accuracy = hist_dict['decoder_siso_2_accuracy']
# decoder3_accuracy = hist_dict['decoder_siso_3_accuracy']
# # val_acc = hist_dict['val_acc']
# print(loss)
# epoch = np.arange(1, epochs + 1)
#
# # plt.semilogy(epoch,val_loss,label='val_loss')
# plt.semilogy(epoch, loss, label='union_loss')
# plt.title('union loss')
# plt.legend(loc=0)
# plt.grid('true')
# plt.xlabel('epochs')
# plt.ylabel('categorical cross-entropy loss')
# plt.show()
#
# plt.semilogy(epoch, decoder0_loss, label='decoder0_loss')
# plt.title('decoder0 loss')
# plt.legend(loc=0)
# plt.grid('true')
# plt.xlabel('epochs')
# plt.ylabel('categorical cross-entropy loss')
# plt.show()
#
# plt.semilogy(epoch, decoder1_loss, label='decoder1_loss')
# plt.title('decoder1 loss')
# plt.legend(loc=0)
# plt.grid('true')
# plt.xlabel('epochs')
# plt.ylabel('categorical cross-entropy loss')
# plt.show()
#
# plt.semilogy(epoch, decoder2_loss, label='decoder2_loss')
# plt.title('decoder2 loss')
# plt.legend(loc=0)
# plt.grid('true')
# plt.xlabel('epochs')
# plt.ylabel('categorical cross-entropy loss')
# plt.show()
#
# plt.semilogy(epoch, decoder3_loss, label='decoder3_loss')
# plt.title('decoder3 loss')
# plt.legend(loc=0)
# plt.grid('true')
# plt.xlabel('epochs')
# plt.ylabel('categorical cross-entropy loss')
# plt.show()




