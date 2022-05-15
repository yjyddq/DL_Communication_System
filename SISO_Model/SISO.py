import os

os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, Flatten, Activation,GRU,LayerNormalization,Embedding,Bidirectional,LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, History, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as KR
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

'''
 --- COMMUNICATION PARAMETERS ---
'''

# 将多少个bit进行打包
k = 1

# Number of symbols
# 交织块的长度
# 一帧的长度为B*B
B = 10
L = B**2

# 编码译码时的中间码率 1/dim
dim = 50

# 最终码率，使用信息位、编码位、交织编码位
n = 3

# Effective Throughput
#  bits per symbol / channel use
R = (1 / n) * (k / 1) # 总码率就是k/n

# Eb/N0 used for training
train_Eb_dB = 2

# Noise Standard Deviation
noise_sigma = np.sqrt(1 / (2 * 10 ** (train_Eb_dB / 10)))


# Number of messages used for training, each size = k*L
batch_size = 64
nb_train_word = batch_size*200

# Probability of burst noise
alpha = 0.05
burst_beta = np.random.binomial(1,alpha,size=(batch_size,L,2*n))
#Set the bursty noise variance
burstyNoise = 1.0
'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
train_data = np.random.randint(low=0, high=2, size=(nb_train_word, L))
# Used as labeled data
label_data = copy.copy(train_data)
train_data = np.reshape(train_data, newshape=(nb_train_word, L, 1))


# 做矩阵乘法 将二进制转换为十进制 shape=(nb_train_word, L, 1)
vec_one_hot = to_categorical(y=train_data, num_classes=2)
# shape=(nb_train_word,L,2**k)
# 将十进制数值转换为one_hot编码
# used as Label data
label_one_hot = copy.copy(vec_one_hot)

'''
 --- NEURAL NETWORKS PARAMETERS ---
'''

early_stopping_patience = 100

epochs = 200

optimizer = Adam(learning_rate=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)
# EarlyStopping 中的patience参数的含义是能够容忍观测指标没有变化的epochs数
# 超过这个patience值那么训练将提前停止
# 我们希望在训练的过程中找到一个最好的点（比如loss最小的点）
# 但是在训练的过程中loss会有抖动，并且训练到一定程度时，loss不降反增
# 所以我们可能需要早点停止训练，比如近几个epochs loss变化很小，就早停
# 而由于loss存在抖动，所以在训练的时候，这个patience的之不能设置的过小
# 否则就会陷入抖动，同理也不能设置的很大，如果很大后期的训练可能会发生过拟合

# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)
# ReduceLROnPlateau params
# factor 学习率下降的速率 new_lr=lr * factor
# patience 当这么多个epochs检测指标还没有改善，那么学习率将会下降
# mode auto max min 希望检测指标变化的方向
# cooldown 在lr减少后，恢复正常之前需要等待的epochs数
# min_lr 学习率下降的下限

# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'DeepTurbo_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)

# Define Power Norm for Tx
def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)  # 2 = I and Q channels
# 这一步操作是进行能量约束
# 这样的话，同相分量和正交分量的能量之和就会<=n channel use

def Norm(x):
    mean = tf.reduce_mean(x)
    var = tf.reduce_mean(tf.square(x-mean))
    return (x-mean)/tf.sqrt(2*var)

# 编码解码单元
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

# 规则交织器
class Se_Regular_Interleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Se_Regular_Interleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(batch_size, B, B))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,0,i*self.part_len],
                         [tf.shape(x)[0],self.part_num,self.part_len])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],1,self.part_len*self.part_num))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([y,tmp],axis=1)
        y = tf.reshape(y,shape=(batch_size,L,1))
        return y

# 解交织器
class Se_Regular_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Se_Regular_DeInterleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(batch_size, B, B))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,i,0],
                         [tf.shape(x)[0],1,tf.shape(x)[-1]])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],self.part_num,self.part_len))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([y,tmp],axis=-1)
        y = tf.reshape(y, shape=(batch_size, L, 1))
        return y

# 逆序交织器
class Re_Regular_Interleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Re_Regular_Interleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,0,i*self.part_len],
                         [tf.shape(x)[0],self.part_num,self.part_len])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],1,self.part_len*self.part_num))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([tmp,y],axis=1)
        return y

# 逆序解交织器
class Re_Regular_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Re_Regular_DeInterleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,i,0],
                         [tf.shape(x)[0],1,tf.shape(x)[-1]])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],self.part_num,self.part_len))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([tmp,y],axis=-1)
        return y

# 伪随机交织器
# 交织的长度为B
# 就是对一帧的数据伪随机打乱
# 在进行伪随机交织时遇到的问题，对tensor进行shuffle后，梯度不能进行传播
# 这是tensorflow的一个问题,官方还并未解决
# 所以采用另一种方案，即对不需要进行梯度传播的indices进行shuffle
# 然后在使用tf.gather对tensor按照indices进行排列
# 这样就能得到伪随机打乱的tensor，并且能进行梯度传播
class Pseudo_random_Interleaver(tf.keras.layers.Layer):
    def __init__(self,L,dim):
        super(Pseudo_random_Interleaver,self).__init__()
        self.L = L # 一帧的长度
        self.dim = dim
    def call(self,x):
        # shape=(batch_size,L,1)
        p = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(batch_size):
            xtmp = tf.slice(x,[i,0,0],[1,self.L,self.dim])
            # shape=(1,L,dim)
            xtmp = tf.reshape(xtmp,shape=(self.L,self.dim))
            np.random.seed(i)
            mshuf = np.random.permutation(mseq)
            ptmp = tf.gather(xtmp,mshuf)
            # shape=(L,dim)
            ptmp = tf.reshape(ptmp,shape=(1,self.L,self.dim))
            # shape=(1,L,dim)
            if i == 0:
                p = ptmp
            else:
                p = tf.concat([p,ptmp],axis=0)
                # shape=(batch_size,L,dim)
        return p

# 伪随机解交织器
# 使用相同的种子，则会产生相同的伪随机数组
# 我们产生一个indices数组，以同样的seed进行随机打乱，在对其进行从小到大
# 进行排序，那么这其中所做的操作，正好也是我们需要对接受序列所做的操作
# 所以我们用排序生成的indices，对接收到的tensor进行排列
# 这就实现了解交织
class Pseudo_random_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,L,dim):
        super(Pseudo_random_DeInterleaver,self).__init__()
        self.L = L # 一帧的长度
        self.dim = dim
    def call(self,x):
        # shape=(batch_size,L,1)
        y = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(batch_size):
            xtmp = tf.slice(x,[i,0,0],[1,self.L,self.dim])
            # shape=(1,L,1)
            xtmp = tf.reshape(xtmp, shape=(self.L,self.dim))
            # shape=(L)
            np.random.seed(i)
            # 将标号按照相同的seed进行打乱
            mshuf = np.random.permutation(mseq)
            # argsort默认按照从小到大排序，返回值为对应元素索引值
            # 即得到恢复mshuf正常顺序的index，同理这也是被打乱bit的恢复index
            indices = tf.argsort(mshuf)
            ytmp = tf.gather(xtmp,indices)
            ytmp = tf.reshape(ytmp,shape=(1,self.L,self.dim))
            # shape=(1,L,1)
            if i == 0:
                y = ytmp
            else:
                y = tf.concat([y,ytmp],axis=0)
                # shape=(batch_size,L,1)
        return y


# 一个batch_size进行交织
class BC_Pseudo_random_Interleaver(tf.keras.layers.Layer):
    def __init__(self, L):
        super(BC_Pseudo_random_Interleaver, self).__init__()
        self.L = L  # 一帧的长度
        self.seed = 0
    def call(self, x):
        # shape=(batch_size,L,1)
        # 种子的循环周期可以设定
        mseq = np.arange(batch_size*L)
        xtmp = tf.reshape(x,shape=(batch_size*L,1))
        # shape=(batch_size*L,1)
        np.random.seed(self.seed)
        mshuf = np.random.permutation(mseq)
        p = tf.gather(xtmp, mshuf)
        # shape=(batch_size*L,1)
        p = tf.reshape(p, shape=(batch_size, L, 1))
        # shape=(batch_size,L,1)
        self.seed += 1
        if self.seed == 64:
            self.seed = 0
        return p

class BC_Pseudo_random_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self, L):
        super(BC_Pseudo_random_DeInterleaver, self).__init__()
        self.L = L  # 一帧的长度
        self.seed = 0
    def call(self, x):
        # shape=(batch_size,L,1)
        mseq = np.arange(batch_size*L)

        xtmp = tf.reshape(x,shape=(batch_size*L,1))
        # shape=(batch_size*L,1)
        np.random.seed(self.seed)
        # 将标号按照相同的seed进行打乱
        mshuf = np.random.permutation(mseq)
        # argsort默认按照从小到大排序，返回值为对应元素索引值
        # 即得到恢复mshuf正常顺序的index，同理这也是被打乱bit的恢复index
        indices = tf.argsort(mshuf)
        y = tf.gather(xtmp, indices)
        y = tf.reshape(y, shape=(batch_size, L, 1))
        # shape=(batch_size,L,1)
        self.seed += 1
        if self.seed == 64:
            self.seed = 0
        return y

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


class Modulation(tf.keras.layers.Layer):
    def __init__(self,k):
        super(Modulation,self).__init__()
        self.k = k
        # 调制是将bit打包，然后映射为星座图上的坐标点
        self.map = coding_unit(dim,1,1,'elu')
        self.modulation = coding_unit(2,1,1,'linear')
    def call(self,x):
        # x shape=(batch_size,L,n)
        xtmp = tf.reshape(x, shape=(batch_size, (L * n) // k, k*dim))
        # k个bit一起打包
        mo = self.map(xtmp)
        # shape = (batch_size, L*n//k ,dim)
        coordinate = self.modulation(mo)
        # shape = (batch_size, L*n//k ,2)
        # 分成k个时隙进行发送
        return coordinate
# 使用one-hot的效果要比1，-1,好
class Encoder(tf.keras.layers.Layer):
    def __init__(self,L):
        super(Encoder,self).__init__()
        self.L = L
        self.codex1_1 = coding_unit(dim,5,1,'elu')
        self.codex1_2 = coding_unit(dim, 5, 1, 'elu')
        self.codex2_1 = coding_unit(dim,5,1,'elu')
        self.codex2_2 = coding_unit(dim, 5, 1, 'elu')
        self.codeI_1 = coding_unit(dim,5,1,'elu')
        self.codeI_2 = coding_unit(dim, 5, 1, 'elu')
        # self.fc1 = Dense(1)
        # self.fc2 = Dense(1)
        # self.fcI = Dense(1)
        self.PRI = Pseudo_random_Interleaver(L,2)
        self.modulation = Modulation(k)
    def call(self,x):
        # shape=(batch_size,L,1)
        e1 = self.codex1_1(x)
        e1 = self.codex1_2(e1)
        # e1 = self.fc1(e1)
        # 可以尝试去掉Dense层，直接将n为特征送到调制器
        e2 = self.codex2_1(x)
        e2 = self.codex2_2(e2)
        # e2 = self.fc2(e2)


        xI = self.PRI(x)
        eI = self.codeI_1(xI)
        eI = self.codeI_2(eI)
        # eI = self.fcI(eI)

        y = tf.concat([e1,e2,eI],axis=-1)
        # shape=(batch_size,L,n*dim)
        y = self.modulation(y)
        # y = self.power_norm(y)
        # # shape = (batch_size, L * n//m, 2)
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
        self.remap = coding_unit(dim,1,1,'elu')
        self.demodulation = coding_unit(k*dim,1,1,'linear')
    def call(self,x):
        # x shape = (batch_size, (L*n)//k, 2)
        rm = self.remap(x)
        # shape = (batch_size, (L*n)//k , dim)
        codeword = self.demodulation(rm)
        # shape = (batch_size, (L*n)//k, k*dim)
        bitseq = tf.reshape(codeword, shape=(batch_size, L, n*dim))
        return bitseq

# 这里的迭代译码完全仿照TurboAE中的方法
# 定义一个迭代网络，网络的层数加深，对这一个网络迭代很多次
# 采用这种模型的效果更好
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
        self.PRI = Pseudo_random_Interleaver(L,dim)
        self.DePRI = Pseudo_random_DeInterleaver(L,dim)
        # self.fc1 = Dense(1)
        # self.fc2 = Dense(1)
    def call(self,dx,de,dI,p):
        tmp = tf.concat([de,p,dx],axis=-1)
        q = self.decodex1(tmp)
        q = self.decodex2(q)
        q = self.decodex3(q)
        q = self.decodex4(q)
        q = self.decodex5(q)
        # q = self.fc1(q)
        # shape=(batch_size, L,1)
        p = self.PRI(q)

        dxI = self.PRI(dx)
        tmp = tf.concat([dI, p, dxI],axis=-1)
        q = self.decodeI1(tmp)
        q = self.decodeI2(q)
        q = self.decodeI3(q)
        q = self.decodeI4(q)
        q = self.decodeI5(q)
        # q = self.fc2(q)
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
        self.p = tf.Variable(tf.random.normal(shape=(batch_size,L,dim)),
                             trainable=True)
    def call(self,x):
        d = self.demodulation(x)
        # shape=(batch_size,L,n)
        de1 = tf.slice(d, [0, 0, 0], [batch_size, L, dim])
        de2 = tf.slice(d, [0, 0, dim],[batch_size,L,dim])
        dI = tf.slice(d, [0, 0, 2*dim],[batch_size,L,dim])
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
    x_real = tf.slice(x,[0,0,0],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    x_imag = tf.slice(x,[0,0,tf.shape(x)[-1]//2],[tf.shape(x)[0],tf.shape(x)[1],tf.shape(x)[-1]//2])
    xc = tf.complex(x_real,x_imag)
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




model_input = Input(batch_shape=(batch_size, L,2), name='input_bits')

e = Encoder(L)(model_input)


e_power = Lambda(normalization)(e)


y_h = AWGN_Channel(noise_sigma)(e_power)

model_output=Decoder(L)(y_h)
# Build System Model
sys_model = Model(model_input, model_output)
encoder = Model(model_input, e)
decoder = Model(y_h,model_output)
# Print Model Architecture
sys_model.summary()


# Compile Model
sys_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


print('starting train the NN...')
start = time.perf_counter()

# TRAINING
mod_history = sys_model.fit(vec_one_hot, vec_one_hot,
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                shuffle=True, # 对一个batch内的数据进行混洗
                                validation_split=0.3, callbacks=[modelcheckpoint,reduce_lr,early_stopping])



end = time.perf_counter()

print('The NN has trained ' + str(end - start) + ' s')


# Plot the Training Loss and Validation Loss
hist_dict = mod_history.history

val_loss = hist_dict['val_loss']
loss = hist_dict['loss']
accuracy = hist_dict['accuracy']
val_accuracy = hist_dict['val_accuracy']
print('loss:',loss)
print('val_loss:',val_loss)

epoch = np.arange(1, epochs + 1)

plt.semilogy(epoch,val_loss,label='val_loss')
plt.semilogy(epoch, loss, label='loss')

plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('Categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'loss')
plt.show()

plt.plot(epoch,accuracy,label='accuracy')
plt.plot(epoch,val_accuracy,label='val_accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'accuracy')
plt.show()
