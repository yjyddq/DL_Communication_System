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

from DL_Communication_System.Coding_Unit.Encoder import Encoder_CNN_PRI
from DL_Communication_System.Coding_Unit.Decoder import Decoder_CNN_PRI
from DL_Communication_System.Power_Norm.power_norm import normalization
from DL_Communication_System.Channel.channel import AWGN_Channel
'''
 --- COMMUNICATION PARAMETERS ---
'''

# bits of symbols
k = 1

# Period of Block Interleaver
B = 10
# Number of symbols
L = B**2

# dimension
dim = 50

# Rcod = 1/n
n = 3

# Effective Throughput
#  bits per symbol / channel use
R = (1 / n) * (k / 1) 

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

train_data = np.reshape(train_data, newshape=(nb_train_word, L, 1))

vec_one_hot = to_categorical(y=train_data, num_classes=2)


'''
 --- NEURAL NETWORKS PARAMETERS ---
'''

early_stopping_patience = 100

epochs = 200

optimizer = Adam(learning_rate=0.001)

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=early_stopping_patience)

# Learning Rate Control
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1,
                              patience=5, min_lr=0.0001)


# Save the best results based on Training Set
modelcheckpoint = ModelCheckpoint(filepath='./' + 'DeepTurbo_' + str(k) + '_' + str(L) + '_' + str(n) + '_' + str(train_Eb_dB) + 'dB' + ' ' + 'AWGN' + '.h5',
                                  monitor='loss',
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=True,
                                  mode='auto', period=1)

model_input = Input(batch_shape=(batch_size, L,1), name='input_bits')

e = Encoder(batch_size,L,dim,k,n)(model_input)

e_power = Lambda(normalization)(e)

y_h = AWGN_Channel(noise_sigma)(e_power)

model_output=Decoder(batch_size,L,dim,k,n)(y_h)


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
