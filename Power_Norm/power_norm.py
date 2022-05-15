from keras import backend as KR
import tensorflow as tf

def normalization(x):
    mean = KR.mean(x ** 2)
    return x / KR.sqrt(2 * mean)

def Norm(x):
    mean = tf.reduce_mean(x)
    var = tf.reduce_mean(tf.square(x-mean))
    return (x-mean)/tf.sqrt(2*var)