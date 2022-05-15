import tensorflow as tf
from tensorflow.keras.layers import Conv1D,BatchNormalization,Activation

# 编码解码单元
class coding_unit(tf.keras.layers.Layer):
    def __init__(self,dim,kernel_size,strides,act):
        super(coding_unit,self).__init__()
        self.conv=Conv1D(filters=dim, strides=strides, kernel_size=kernel_size,padding='same')
        self.BN=BatchNormalization()
        self.act=Activation(act)
    def call(self,x):
        y=self.conv(x)
        y=self.BN(y)
        y=self.act(y)
        return y

