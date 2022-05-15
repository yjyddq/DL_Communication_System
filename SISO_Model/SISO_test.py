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
test_data = np.random.randint(low=0, high=2, size=(num_of_sym, L))

test_data = np.reshape(test_data, newshape=(num_of_sym, L, 1))

vec_one_hot = to_categorical(y=test_data, num_classes=2)


print('start simulation ...' + str(k) + '_' + str(L) + '_' + str(n))

'''
 --- DEFINE THE Neural Network(NN) ---
'''

# Eb_N0 in dB
for Eb_N0_dB in range(0, 16):
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
    
#     #  Plot constellation
#     path = './' + 'AWGN_1_100_3_2dB_result/' + 'DeepTurbo_encoder_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
    
#     # encoder = Model(model_input, e)
#     encoder.load_weights(path)
#     awgn=AWGN_Channel(noise_sigma)
#     # rayleigh = Rayleigh_Channel(noise_sigma)
#     C=[]
#     with tf.device("/cpu:0"):
#         for i in range(100):
#             c = encoder(train_data[0+i*64:64+i*64])
#             c = awgn(c)
#             if i == 0:
#                 C = c
#             else:
#                 C = tf.concat([C,c],axis=1)
#     fig = plt.figure(1)
#     plt.title('Constellation k=' + str(k) + ' test at ' + str(Eb_N0_dB)+' Rayleigh')
#     plt.xlim(-3.5, 3.5)
#     plt.ylim(-3.5, 3.5)
#     plt.plot(C[1, :, 0], C[1, :, 1], 'ro')
    
#     plt.grid(True)
#     plt.savefig('./' + 'Constellation_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN')
#     plt.show()

    
'''
PLOTTING
'''
# Print BER

print(Vec_Eb_N0, '\n', Bit_error_rate)

with open('BER_DeepTurbo_'+str(k)+'_'+str(n)+'_'+str(L)+'train at'+str(train_Eb_dB)+'dB'+'_AWGN'+'.txt', 'w') as f:
    for i in range(11):
        print(Vec_Eb_N0[i], 'dB ','BER: ',Bit_error_rate[i] ,'\n', file=f)
f.closed

# Plot BER Figure
plt.semilogy(Vec_Eb_N0, Bit_error_rate)
label = ['TurboAE-MOD'+str(k) + '_' + str(L) + '_' + str(n)]
plt.legend(label, loc=0)
plt.xlabel('Eb/N0')
plt.ylabel('BER')
plt.title('AWGN'+'_'+str(k) + '_' + str(n)+'_'+str(L))
plt.grid('true')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'BER')
plt.show()

