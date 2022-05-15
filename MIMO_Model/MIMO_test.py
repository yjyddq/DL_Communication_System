# test MIMO OFDM
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Lambda, BatchNormalization, Input, Conv1D, TimeDistributed, LeakyReLU, Flatten, Activation,Embedding,LocallyConnected1D
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as KR
import copy
import tensorflow as tf


'''
 --- COMMUNICATION PARAMETERS ---
'''

# number of information bits
k = 1

# codeword Length
L = 100

# Channel use
n = 3

dim = 50

# User number
m = 4

# Effective Throughput
#  (bits per symbol)*( number of symbols) / channel use
R = k/n

# Eb/N0 used for training(load_weights)
train_Eb_dB = 2

# Number of messages used for test, each size = k*L
batch_size = 64
num_of_sym = batch_size*1000

# Initial Vectors
Vec_Eb_N0 = []
Bit_error_rate0 = []
Bit_error_rate1 = []
Bit_error_rate2 = []
Bit_error_rate3 = []

'''
 --- GENERATING INPUT DATA ---
'''

# Generate training binary Data
# User one
# Generate training binary Data
# User one
train_data0 = np.random.randint(low=0, high=2, size=(num_of_sym, L,1))
vec_one_hot0 = to_categorical(y=train_data0, num_classes=2)

# User two
train_data1 = np.random.randint(low=0, high=2, size=(num_of_sym, L,1))
vec_one_hot1 = to_categorical(y=train_data1, num_classes=2)

# User three
train_data2 = np.random.randint(low=0, high=2, size=(num_of_sym, L,1))
vec_one_hot2 = to_categorical(y=train_data2, num_classes=2)

# User four
train_data3 = np.random.randint(low=0, high=2, size=(num_of_sym, L,1))
vec_one_hot3 = to_categorical(y=train_data3, num_classes=2)


print('start simulation ...' + str(k) + '_' + str(L)+'_'+str(n))


'''
 --- DEFINE THE Neural Network(NN) ---
'''

# Eb_N0 in dB
for Eb_N0_dB in range(0,16):

    # Noise Sigma at this Eb
    noise_sigma = np.sqrt(1 / (2 * R * 10 ** (Eb_N0_dB / 10)))

    # Define Encoder Layers (Transmitter)
    model_input0 = Input(batch_shape=(batch_size, L,1),name='inputs0')
    model_input1 = Input(batch_shape=(batch_size, L,1),name='inputs1')
    model_input2 = Input(batch_shape=(batch_size, L,1),name='inputs2')
    model_input3 = Input(batch_shape=(batch_size, L,1),name='inputs3')


    e0 = EncoderSISO(L,name='encoder0')(model_input0)
    e1 = EncoderSISO(L,name='encoder1')(model_input1)
    e2 = EncoderSISO(L,name='encoder2')(model_input2)
    e3 = EncoderSISO(L,name='encoder3')(model_input3)

    f = OFDM(m,name='OFDM')(e0,e1,e2,e3)

    y_h = AWGN_four_Channel(noise_sigma,name='Channel')(f)


    d0,d1,d2,d3=DeOFDM(m,name='DeOFDM')(y_h)

    model_output0 = DecoderSISO(L,name='decoder0')(d0)
    model_output1 = DecoderSISO(L,name='decoder1')(d1)
    model_output2 = DecoderSISO(L,name='decoder2')(d2)
    model_output3 = DecoderSISO(L,name='decoder3')(d3)

    # Build System Model
    sys_model = Model(inputs=[model_input0,model_input1,model_input2,model_input3], outputs=[model_output0,model_output1,model_output2,model_output3])

    # Load Weights from the trained NN
    sys_model.load_weights('../input/mimomodel/' + 'model_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                       by_name=False)


    '''
    RUN THE NN
    '''

    decoder_output = sys_model.predict([train_data0,train_data1,train_data2,train_data3], batch_size=batch_size)


    '''
     --- CALULATE BLER ---
    '''

    # Decode One-Hot vector
    # Compute User0 BER
    position0 = np.argmax(decoder_output[0], axis=2)
    tmp0 = np.reshape(position0,newshape=train_data0.shape)

    error_rate0 = np.mean(np.not_equal(train_data0,tmp0))
    # Compute User1 BER
    position1 = np.argmax(decoder_output[1], axis=2)
    tmp1 = np.reshape(position1, newshape=train_data1.shape)

    error_rate1 = np.mean(np.not_equal(train_data1, tmp1))
    # Compute User2 BER
    position2 = np.argmax(decoder_output[2], axis=2) 
    tmp2 = np.reshape(position2, newshape=train_data2.shape)

    error_rate2 = np.mean(np.not_equal(train_data2, tmp2))
    # Compute User3 BER
    position3 = np.argmax(decoder_output[3], axis=2)
    tmp3 = np.reshape(position3, newshape=train_data3.shape)

    error_rate3 = np.mean(np.not_equal(train_data3, tmp3))

    print('Eb/N0 = ', Eb_N0_dB)
    print('User zero BLock Error Rate = ', error_rate0)
    print('User one BLock Error Rate = ', error_rate1)
    print('User two BLock Error Rate = ', error_rate2)
    print('User three BLock Error Rate = ', error_rate3)

    print('\n')

    # Store The Results
    Vec_Eb_N0.append(Eb_N0_dB)
    Bit_error_rate0.append(error_rate0)
    Bit_error_rate1.append(error_rate1)
    Bit_error_rate2.append(error_rate2)
    Bit_error_rate3.append(error_rate3)

'''
PLOTTING
'''
# Print BER
# print(Bit_error_rate)

# print(Vec_Eb_N0, '\n', Bit_error_rate1)

with open('BLER_model_four_users_'+str(k)+'_'+str(n)+'_'+str(L)+'train at'+str(train_Eb_dB)+'dB'+'_AWGN'+'.txt', 'w') as f:
    for i in range(16):
        print(Vec_Eb_N0[i], 'dB '+'\n','User0: ',Bit_error_rate0[i],'\n',
                                  'User1: ',Bit_error_rate1[i],'\n',
                                  'User2: ', Bit_error_rate2[i],'\n',
                                  'User3: ', Bit_error_rate3[i],'\n', file=f)
f.closed

# Plot BER Figure
plt.semilogy(Vec_Eb_N0, Bit_error_rate0,
             Vec_Eb_N0, Bit_error_rate1,
             Vec_Eb_N0, Bit_error_rate2,
             Vec_Eb_N0, Bit_error_rate3)
label = ['User0 '+str(k) + '_' + str(L),'User1 '+str(k) + '_' + str(L),
         'User2 '+str(k) + '_' + str(L),'User3 '+str(k) + '_' + str(L)]
plt.legend(label, loc=0)
plt.xlabel('Eb/N0')
plt.ylabel('BLER')
plt.title(str(k) + '_' + str(n)+'_'+str(L))
plt.grid('true')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'BER')
plt.show()
