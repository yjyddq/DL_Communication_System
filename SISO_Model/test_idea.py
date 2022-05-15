# 统计在使用循环卷积前后
# 边缘数据的误码率性能的差异
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, LeakyReLU, \
    Flatten, Activation, Conv1DTranspose, LSTM, LayerNormalization, Bidirectional
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as KR
import copy
import tensorflow as tf

'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 1

# Number of symbols
B = 10
L = B ** 2

dim = 50

# Channel Use
n = 3

# Effective Throughput
#  bits per symbol / channel use
R = k / n

# Eb/N0 used for training
train_Eb_dB = 2

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * 10 ** (train_Eb_dB / 10)))

# Number of messages used for test, each size = k*L
batch_size = 64
num_of_sym = batch_size * 1000

# Probability of burst noise
alpha = 0.05
burst_beta = np.random.binomial(1, alpha, size=(batch_size, L, 2 * n))
# Set the bursty noise variance
burstyNoise = 1.0

# Initial Vectors
Vec_Eb_N0 = []
Bit_error_rate = []

'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
train_data = np.random.randint(low=0, high=2, size=(num_of_sym, L))
# Used as labeled data
label_data = copy.copy(train_data)
train_data = np.reshape(train_data, newshape=(num_of_sym, L, 1))

vec_one_hot = to_categorical(y=train_data, num_classes=2)


# Define Power Norm for Tx
def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels

class Circular_Conv(tf.keras.layers.Layer):
    def __init__(self,filters,kernel_size,strides,padding):
        super(Circular_Conv,self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = Conv1D(filters=filters,kernel_size=kernel_size,
                           strides=strides,padding='valid')
    def call(self,x):
        # x shape=(batch_size,L,dim)
        y = x
        if self.padding == 'valid':
            y = self.conv(x)
        elif self.padding == 'same':
            pad_window = ((self.strides-1)*tf.shape(x)[1]-self.strides+self.kernel_size)//2
            head_pad = tf.slice(x,[0,L-pad_window,0],[batch_size,pad_window,tf.shape(x)[-1]])
            tail_pad = tf.slice(x,[0,0,0],[batch_size,pad_window,tf.shape(x)[-1]])
            x_pad = tf.concat([head_pad,x,tail_pad],axis=1)
            y = self.conv(x_pad)
        return y

# 编码解码单元
class coding_unit(tf.keras.layers.Layer):
    def __init__(self,dim,kernel_size,strides,act,padding):
        super(coding_unit,self).__init__()
        self.dim=dim
        self.conv=Circular_Conv(filters=dim, strides=strides, kernel_size=kernel_size,padding=padding)
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
        self.map = coding_unit(dim,1,1,'elu','valid')
        self.modulation = coding_unit(2,1,1,'linear','valid')
    def call(self,x):
        # x shape=(batch_size,L,n)
        xtmp = tf.reshape(x, shape=(batch_size, (L * n) // k, k))
        # k个bit一起打包
        mo = self.map(xtmp)
        # shape = (batch_size, L*n//k ,dim)
        coordinate = self.modulation(mo)
        # shape = (batch_size, L*n//k , 2)
        # 分成k个时隙进行发送
        return coordinate

class Encoder(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Encoder,self).__init__()
        self.L = L
        self.codex1_1 = coding_unit(dim,5,1,'elu','same')
        # self.codex1_2 = coding_unit(dim, 5, 1, 'elu','same')
        self.codex2_1 = coding_unit(dim,5, 1, 'elu','same')
        # self.codex2_2 = coding_unit(dim, 5, 1, 'elu','same')
        self.codeI1 = coding_unit(dim,5,1,'elu','same')
        # self.codeI2 = coding_unit(dim, 5, 1, 'elu','same')
        self.fc1 = Dense(1)
        self.fc2 = Dense(1)
        self.fcI = Dense(1)
        self.PRI = Pseudo_random_Interleaver(L)
        self.modulation = Modulation(k)
    def call(self,x):
        x = 2 * x - 1
        # shape=(batch_size,L,1)
        e1 = self.codex1_1(x)
        # e1 = self.codex1_2(e1)
        e1 = self.fc1(e1)

        e2 = self.codex2_1(x)
        # e2 = self.codex2_2(e2)
        e2 = self.fc2(e2)

        xI = self.PRI(x)
        eI = self.codeI1(xI)
        # eI = self.codeI2(eI)
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
        base_config = super(Encoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DeModulation(tf.keras.layers.Layer):
    def __init__(self,n):
        super(DeModulation,self).__init__()
        self.n = n
        # 调制是将bit打包，然后映射为星座图上的坐标点
        self.remap = coding_unit(dim,1,1,'elu','valid')
        self.demodulation = coding_unit(k,1,1,'linear','valid')
    def call(self,x):
        # x shape = (batch_size, (L*n)//k, 2)
        rm = self.remap(x)
        # shape = (batch_size, (L*n)//k , dim)
        codeword = self.demodulation(rm)
        # shape = (batch_size, (L*n)//k, k)
        bitseq = tf.reshape(codeword, shape=(batch_size, L, n))
        return bitseq

class Iteration_dec(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Iteration_dec,self).__init__()
        self.L = L
        self.decodex1 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodex2 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodex3 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodex4 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodex5 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodeI1 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodeI2 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodeI3 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodeI4 = coding_unit(dim, 5, 1, 'elu','same')
        self.decodeI5 = coding_unit(dim, 5, 1, 'elu','same')
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


# 伪随机交织
class Decoder(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Decoder,self).__init__()
        self.L = L
        self.iter = Iteration_dec(L)
        self.demodulation = DeModulation(n)
        self.dec_out = Conv1D(filters=2,strides=1,kernel_size=1,activation='softmax')
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,1)),
                             trainable=True)
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
        base_config = super(Decoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AWGN_Channel(tf.keras.layers.Layer):
    def __init__(self,noise_sigma):
        super(AWGN_Channel,self).__init__()
        self.noise_sigma=noise_sigma
    def call(self,x):
        w = KR.random_normal(KR.shape(x), mean=0.0, stddev=self.noise_sigma)
        y = x + w
        return y


# 将I-Q 2位实数转换为一位的复数
def real_convert_to_complex(x):
    # shape=(batch_size,L,2)
    xc = tf.expand_dims(tf.complex(x[:,:,0],x[:,:,1]),axis=-1)
    # shape=(batch_size,L,1)
    return xc

# 将I-Q 2位实数转换为一位的复数
def complex_convert_to_real(xc):
    # shape=(batch_size,L,m)
    x_real = tf.math.real(xc)
    x_imag = tf.math.imag(xc)
    x = tf.concat([x_real,x_imag],axis=-1)
    # shape=(batch_size,L,2)
    return x

class ISI(tf.keras.layers.Layer):
    def __init__(self,W):
        super(ISI,self).__init__()
        self.W = W # ISI窗口，即t时刻的码元受前W个时刻的码元影响
    def call(self,x):
        y_isi=[]
        xc = real_convert_to_complex(x)
        hc = real_convert_to_complex(KR.random_normal(KR.shape(x), mean=0.0, stddev=np.sqrt(1 / 2)))
        isic = tf.multiply(hc, xc)
        isi = complex_convert_to_real(isic)
        for i in range(L*n//k):
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

# 缓解ISI可以使用RNN，因为一帧的第一个时隙（一帧的开始无码间串扰）
# 所以由第一个码元可以帮助后续码元的恢复
# 加入RNN单元会比不加RNN单元要好
# 解决码间串扰问题我们可以在调制模块前再加入一个RNN结构
class DeISI(tf.keras.layers.Layer):
    def __init__(self,dim):
        super(DeISI,self).__init__()
        self.dim = dim
        self.diff1 = LSTM(dim,return_sequences=True)
        self.diff2 = LSTM(dim,return_sequences=True)
        self.dense = Dense(2)
    def call(self,x):
        df1 = self.diff1(x)
        # shape=(batch_size,L*n//k,dim)
        df2 = self.diff2(df1)
        # shape=(batch_size,L*n//k,dim)
        de_isi = self.dense(df2)
        # shape=(batch_size,L*n//k,2)
        return de_isi
    def get_config(self):
        config = {
            'dim':
                self.dim
        }
        base_config = super(DeISI, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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

print('start simulation ...' + str(k) + '_' + str(L) + '_' + str(n))

'''
 --- DEFINE THE Neural Network(NN) ---
'''
# W = 1
# Eb_N0 in dB
for Eb_N0_dB in range(0, 11):
    # Noise Sigma at this Eb
    noise_sigma = np.sqrt(1 / (2 * 10 ** (Eb_N0_dB / 10)))

    # Define Encoder Layers (Transmitter)
    model_input = Input(batch_shape=(batch_size, L, 1), name='input_bits')

    e = Encoder(L)(model_input)

    e_power = Lambda(normalization)(e)

    # e_isi = ISI(W)(e_power)

    y_h = Rayleigh_Channel(noise_sigma)(e_power)


    model_output = Decoder(L)(y_h)
    # Build System Model
    sys_model = Model(model_input, model_output)


    # Load Weights from the trained NN
    sys_model.load_weights(r'E:\pycharm\communication\Memory_code\Turbo Rayleigh\5dB\DeepTurbo_1_100_3_5dB Rayleigh.h5')

    '''
    RUN THE NN
    '''

    # RUN Through the Model and get output

    decoder_output = sys_model.predict(train_data, batch_size=batch_size)
        # shape=(num_of_sym,L,1)

    '''
     --- CALULATE BLER ---
    '''

    # Decode One-Hot vector
    position = np.argmax(decoder_output, axis=2)
    tmp = np.reshape(position, newshape=train_data.shape)

    error_rate = np.mean(np.not_equal(train_data, tmp))

    print('Eb/N0 = ', Eb_N0_dB)
    print('BLock Error Rate = ', error_rate)

    print('\n')

    # Store The Results
    Vec_Eb_N0.append(Eb_N0_dB)
    Bit_error_rate.append(error_rate)
    #
    # #  Plot constellation
    # path = './' + 'AWGN_1_100_3_2dB_result/' + 'DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
    #
    # # encoder = Model(model_input, e)
    # encoder.load_weights(path)
    # awgn=AWGN_Channel(noise_sigma)
    # # rayleigh = Rayleigh_Channel(noise_sigma)
    # C=[]
    # with tf.device("/cpu:0"):
    #     for i in range(100):
    #         c = encoder(train_data[0+i*64:64+i*64])
    #         c = awgn(c)
    #         if i == 0:
    #             C = c
    #         else:
    #             C = tf.concat([C,c],axis=1)
    # fig = plt.figure(1)
    # plt.title('Constellation k=' + str(k) + ' test at ' + str(Eb_N0_dB)+' Rayleigh')
    # plt.xlim(-3.5, 3.5)
    # plt.ylim(-3.5, 3.5)
    # plt.plot(C[1, :, 0], C[1, :, 1], 'ro')
    #
    # plt.grid(True)
    # plt.savefig('./' + 'Constellation_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN')
    # plt.show()

'''
PLOTTING
'''
# Print BER


print(Vec_Eb_N0, '\n', Bit_error_rate)

with open('BER_DeepTurbo_'+str(k)+'_'+str(n)+'_'+str(L)+'train at'+str(train_Eb_dB)+'dB'+'_AWGN'+'.txt', 'w') as f:
    for i in range(11):
        print(Vec_Eb_N0[i], 'dB ','BER: ',Bit_error_rate[i] ,'\n', file=f)
f.closed

# Plot BLER Figure
plt.semilogy(Vec_Eb_N0, Bit_error_rate)
label = ['TurboAE'+str(k) + '_' + str(L) + '_' + str(n)]
plt.legend(label, loc=0)
plt.xlabel('Eb/N0')
plt.ylabel('BER')
plt.title(str(k) + '_' + str(n)+'_'+str(L))
plt.grid('true')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'BER')
plt.show()

model_input = Input(batch_shape=(batch_size, L, 1), name='input_bits')

e = Encoder(L)(model_input)

e_power = Lambda(normalization)(e)

y_h = AWGN_Channel(noise_sigma)(e_power)

model_output = Decoder(L)(y_h)
# Build System Model
sys_model = Model(model_input, model_output)
# encoder = Model(model_input, e)

# Load Weights from the trained NN
sys_model.load_weights('./'+'Circ_CNN/kernel-size5/'+'Circ_CNN_' + str(k) + '_' + str(L) + '_'
                   + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' '
                   + 'AWGN' + '.h5',
                   by_name=False)


'''
RUN THE NN
'''

# RUN Through the Model and get output
with tf.device("/cpu:0"):
    decoder_output = sys_model.predict(train_data, batch_size=batch_size)

position = np.argmax(decoder_output, axis=2)
tmp = np.reshape(position,newshape=train_data.shape)
# shape(num_of_sym,L,1)
error_rate = np.mean(np.not_equal(train_data,tmp),axis=0)

error_rate = np.squeeze(error_rate,axis=-1)

location = np.arange(1,L+1)
plt.plot(location, error_rate,'ko-')
plt.xlabel('Position')
plt.ylabel('BER')
plt.title(str(k) + '_' + str(n) + '_' + str(L))
plt.grid('true')
plt.savefig('./' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'diff_position_BER_Circ_CNN')
plt.show()



# model_input = Input(batch_shape=(batch_size, L, 1), name='input_bits')
#
# e = Encoder(L)(model_input)
#
# e_power = Lambda(normalization)(e)
#
# y_h = AWGN_Channel(noise_sigma)(e_power)
#
# model_output = Decoder(L)(y_h)
#
# encoder = Model(model_input, e)
# #  Plot constellation
# path = './' + 'paper_iter/1_100_3_2dB_AWGN/' + 'DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
#
# # encoder = Model(model_input, e)
# encoder.load_weights(path)
# awgn=AWGN_Channel(noise_sigma)
# # rayleigh = Rayleigh_Channel(noise_sigma)
# C=[]
# with tf.device("/cpu:0"):
#     for i in range(1000):
#         c = encoder(train_data[0+i*64:64+i*64])
#         c = Lambda(normalization)(c)
#         # c = awgn(c)
#         if i == 0:
#             C = c
#         else:
#             C = tf.concat([C,c],axis=1)
# fig = plt.figure(1)
# plt.title('Distribution')
# # plt.xlim(-3.5, 3.5)
# # plt.ylim(-3.5, 3.5)
# # plt.plot(C[1, :, 0], 'ro')
# amplitude = tf.sqrt(tf.square(C[:,:,0])+tf.square(C[:,:,1]))
# amplitude = np.reshape(amplitude,[-1])
# print(amplitude.shape)
# plt.hist(amplitude,bins=100)
# plt.grid(True)
# plt.savefig('./' + 'Distribution' + ' ' +'AWGN')
# plt.show()