import tensorflow as tf
from tensorflow.keras.layers import Conv1D

class Circular_Conv(tf.keras.layers.Layer):
    def __init__(self,batch_size,L,filters,kernel_size,strides,padding):
        super(Circular_Conv,self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.conv = Conv1D(filters=filters,kernel_size=kernel_size,
                           strides=strides,padding='valid')
    def call(self,x):
        # x shape=(batch_size,L,dim)
        y = []
        if self.padding == 'valid':
            y = self.conv(x)
        elif self.padding == 'same':
            pad_window = ((self.strides-1)*tf.shape(x)[1]-self.strides+self.kernel_size)//2
            head_pad = tf.slice(x,[0,self.L-pad_window,0],[self.batch_size,pad_window,tf.shape(x)[-1]])
            tail_pad = tf.slice(x,[0,0,0],[self.batch_size,pad_window,tf.shape(x)[-1]])
            x_pad = tf.concat([head_pad,x,tail_pad],axis=1)
            y = self.conv(x_pad)
        return y