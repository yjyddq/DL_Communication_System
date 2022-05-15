import tensorflow as tf


class Se_Block_Interleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Se_Block_Interleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], self.part_num, self.part_num))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,0,i*self.part_len],
                         [tf.shape(x)[0],self.part_num,self.part_len])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],1,self.part_len*self.part_num))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([y,tmp],axis=1)
        y = tf.reshape(y,shape=(tf.shape(x)[0],self.part_num*self.part_num,1))
        return y


class Se_Block_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Se_Block_DeInterleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], self.part_num, self.part_num))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,i,0],
                         [tf.shape(x)[0],1,tf.shape(x)[-1]])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],self.part_num,self.part_len))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([y,tmp],axis=-1)
        y = tf.reshape(y, shape=(tf.shape(x)[0], self.part_num*self.part_num, 1))
        return y


class Re_Block_Interleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Re_Block_Interleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], self.part_num, self.part_len))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,0,i*self.part_len],
                         [tf.shape(x)[0],self.part_num,self.part_len])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],1,self.part_len*self.part_num))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([tmp,y],axis=1)
        y = tf.reshape(y, shape=(tf.shape(x)[0], self.part_num * self.part_num, 1))
        return y


class Re_Block_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self,part_len,part_num):
        super(Re_Block_DeInterleaver,self).__init__()
        self.part_len=part_len
        self.part_num=part_num
    def call(self,x):
        x = tf.reshape(x, shape=(tf.shape(x)[0], self.part_num, self.part_len))
        y=[]
        for i in range(self.part_num):
            tmp=tf.slice(x,[0,i,0],
                         [tf.shape(x)[0],1,tf.shape(x)[-1]])
            tmp=tf.reshape(tmp,(tf.shape(x)[0],self.part_num,self.part_len))
            if i == 0:
                y=tmp
            else:
                y=tf.concat([tmp,y],axis=-1)
        y = tf.reshape(y, shape=(tf.shape(x)[0], self.part_num * self.part_num, 1))
        return y
