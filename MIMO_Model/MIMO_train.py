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

from DL_Communication_System.Coding_Unit.Encoder import Encoder_CNN_PRI
from DL_Communication_System.Coding_Unit.Decoder import Decoder_CNN_PRI
from DL_Communication_System.Channel.channel import OFDM,DeOFDM,AWGN_four_Channel
from DL_Communication_System.Power_Norm.power_norm import normalization
'''
 --- COMMUNICATION PARAMETERS ---
'''

# Bits per Symbol
k = 1

# Number of symbols
L = 100

# Channel Use
n = 3

# dimension
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

model_input0 = Input(batch_shape=(batch_size, L,1),name='inputs0')
model_input1 = Input(batch_shape=(batch_size, L,1),name='inputs1')
model_input2 = Input(batch_shape=(batch_size, L,1),name='inputs2')
model_input3 = Input(batch_shape=(batch_size, L,1),name='inputs3')

e0 = Encoder_CNN_PRI(batch_size,L,dim,k,n,name='encoder0')(model_input0)
e1 = Encoder_CNN_PRI(batch_size,L,dim,k,n,name='encoder1')(model_input1)
e2 = Encoder_CNN_PRI(batch_size,L,dim,k,n,name='encoder2')(model_input2)
e3 = Encoder_CNN_PRI(batch_size,L,dim,k,n,name='encoder3')(model_input3)

f = OFDM(m,name='OFDM')(e0,e1,e2,e3)

y_h = AWGN_four_Channel(noise_sigma,name='Channel')(f)

d0,d1,d2,d3=DeOFDM(m,name='DeOFDM')(y_h)

model_output0 = Decoder_CNN_PRI(batch_size,L,dim,k,n,name='decoder0')(d0)
model_output1 = Decoder_CNN_PRI(batch_size,L,dim,k,n,name='decoder1')(d1)
model_output2 = Decoder_CNN_PRI(batch_size,L,dim,k,n,name='decoder2')(d2)
model_output3 = Decoder_CNN_PRI(batch_size,L,dim,k,n,name='decoder3')(d3)

# Build System Model
sys_model = Model([model_input0,model_input1,model_input2,model_input3], [model_output0,model_output1,model_output2,model_output3])


# encoder0 = Model(model_input0,e0)
# encoder1 = Model(model_input1,e1)
# encoder2 = Model(model_input2,e2)
# encoder3 = Model(model_input3,e3)
# decoder0 = Model(d0,model_output0)
# decoder1 = Model(d1,model_output1)
# decoder2 = Model(d2,model_output2)
# decoder3 = Model(d3,model_output3)


# Print Model Architecture
sys_model.summary()

# SISO Encoder Layer
encoder0 = sys_model.layers[4]
encoder1 = sys_model.layers[5]
encoder2 = sys_model.layers[6]
encoder3 = sys_model.layers[7]
decoder0 = sys_model.layers[-4]
decoder1 = sys_model.layers[-3]
decoder2 = sys_model.layers[-2]
decoder3 = sys_model.layers[-1]



# Define SISO Model to Load SISO Weights
model_input = Input(batch_shape=(batch_size, L,1), name='input_bits')

e = Encoder_CNN_PRI(batch_size,L,dim,k,n)(model_input)

e_power = Lambda(normalization)(e)

y_h = AWGN_Channel(noise_sigma)(e_power)


model_output = Decoder_CNN_PRI(batch_size,L,dim,k,n)(y_h)

# Build SISO Model
Turbo_model = Model(model_input, model_output)
Turbo_path = './DeepTurbo_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
Turbo_model.load_weights(Turbo_path)

# Load SISO Encoder Weights
encoder0.set_weights(Turbo_model.layers[1].get_weights())
encoder1.set_weights(Turbo_model.layers[1].get_weights())
encoder2.set_weights(Turbo_model.layers[1].get_weights())
encoder3.set_weights(Turbo_model.layers[1].get_weights())
# Load SISO Decoder Weights
decoder0.set_weights(Turbo_model.layers[4].get_weights())
decoder1.set_weights(Turbo_model.layers[4].get_weights())
decoder2.set_weights(Turbo_model.layers[4].get_weights())
decoder3.set_weights(Turbo_model.layers[4].get_weights())


def custom_loss(real,pred):
    L1 = tf.reduce_mean(-tf.reduce_sum(real[0] * tf.math.log(pred[0])))
    L2 = tf.reduce_mean(-tf.reduce_sum(real[1] * tf.math.log(pred[1])))
    L3 = tf.reduce_mean(-tf.reduce_sum(real[2] * tf.math.log(pred[2])))
    L4 = tf.reduce_mean(-tf.reduce_sum(real[3] * tf.math.log(pred[3])))
    alpha1 = L1 / (L1 + L2 + L3 + L4)
    alpha2 = L2 / (L1 + L2 + L3 + L4)
    alpha3 = L3 / (L1 + L2 + L3 + L4)
    alpha4 = L4 / (L1 + L2 + L3 + L4)
    L = alpha1 * L1 + alpha2 * L2 + alpha3 * L3 + alpha4 * L4
    return L
   
# Compile MIMO Model
sys_model.compile(optimizer=optimizer, loss=custom_loss, metrics=['accuracy'])


print('starting train the NN...')
start = time.perf_counter()

# Training MIMO
mod_history = sys_model.fit([train_data0,train_data1,train_data2,train_data3], [vec_one_hot0,vec_one_hot1,vec_one_hot2,vec_one_hot3],
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=1,
                                shuffle=True,
                                validation_split=0.3, callbacks=[modelcheckpoint])

# encoder0_path='./' + 'encoder0_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder0.save(encoder0_path)

# encoder1_path='./' + 'encoder1_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder1.save(encoder1_path)

# encoder2_path='./' + 'encoder2_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder2.save(encoder2_path)

# encoder3_path='./' + 'encoder3_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder3.save(encoder3_path)

# decoder0_path='./' + 'decoder0_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# decoder0.save(decoder0_path)

# decoder1_path='./' + 'decoder1_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder1.save(decoder1_path)

# decoder2_path='./' + 'decoder2_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder2.save(decoder2_path)

# decoder3_path='./' + 'decoder3_four_users_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5'
# encoder3.save(decoder3_path)

end = time.perf_counter()

print('The NN has trained ' + str(end - start) + ' s')


# Plot the Training Loss and Validation Loss
hist_dict = mod_history.history

# val_loss = hist_dict['val_loss']
loss = hist_dict['loss']
decoder0_loss= hist_dict['decoder0_loss']
decoder1_loss= hist_dict['decoder1_loss']
decoder2_loss= hist_dict['decoder2_loss']
decoder3_loss= hist_dict['decoder3_loss']
decoder0_accuracy = hist_dict['decoder0_accuracy']
decoder1_accuracy = hist_dict['decoder1_accuracy']
decoder2_accuracy = hist_dict['decoder2_accuracy']
decoder3_accuracy = hist_dict['decoder3_accuracy']

epoch = np.arange(1, epochs + 1)

# plt.semilogy(epoch,val_loss,label='val_loss')
plt.semilogy(epoch, loss, label='union_loss')
plt.title('union loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'union_loss')
plt.show()

plt.semilogy(epoch, decoder0_loss, label='decoder0_loss')
plt.title('decoder0 loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder0_loss')
plt.show()

plt.semilogy(epoch, decoder1_loss, label='decoder1_loss')
plt.title('decoder1 loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder1_loss')
plt.show()

plt.semilogy(epoch, decoder2_loss, label='decoder2_loss')
plt.title('decoder2 loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder2_loss')
plt.show()

plt.semilogy(epoch, decoder3_loss, label='decoder3_loss')
plt.title('decoder3 loss')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('categorical cross-entropy loss')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder3_loss')
plt.show()

plt.semilogy(epoch, decoder0_accuracy, label='decoder0_accuracy')
plt.title('decoder0_accuracy')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder0_accuracy')
plt.show()

plt.semilogy(epoch, decoder1_accuracy, label='decoder1_accuracy')
plt.title('decoder1_accuracy')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder1_accuracy')
plt.show()

plt.semilogy(epoch, decoder2_accuracy, label='decoder2_accuracy')
plt.title('decoder2_accuracy')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder2_accuracy')
plt.show()

plt.semilogy(epoch, decoder3_accuracy, label='decoder3_accuracy')
plt.title('decoder3_accuracy')
plt.legend(loc=0)
plt.grid('true')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('./'+str(k)+ '_' + str(L) + '_' + str(n)+ '_' + str(train_Eb_dB) + 'dB'+' '+'decoder3_accuracy')
plt.show()




