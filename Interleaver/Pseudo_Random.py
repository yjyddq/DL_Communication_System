import tensorflow as tf
import numpy as np


# 伪随机交织器
class Pseudo_random_Interleaver(tf.keras.layers.Layer):
    def __init__(self, batch_size, L, dim):
        super(Pseudo_random_Interleaver, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim

    def call(self, x):
        # shape=(batch_size,L,dim)
        p = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(self.batch_size):
            xtmp = tf.slice(x, [i, 0, 0], [1, self.L, self.dim])
            # shape=(1,L,dim)
            xtmp = tf.reshape(xtmp, shape=(self.L, self.dim))
            np.random.seed(i)
            mshuf = np.random.permutation(mseq)
            ptmp = tf.gather(xtmp, mshuf)
            # shape=(L,dim)
            ptmp = tf.reshape(ptmp, shape=(1, self.L, self.dim))
            # shape=(1,L,dim)
            if i == 0:
                p = ptmp
            else:
                p = tf.concat([p, ptmp], axis=0)
                # shape=(batch_size,L,dim)
        return p


# 伪随机解交织器
class Pseudo_random_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self, batch_size, L, dim):
        super(Pseudo_random_DeInterleaver, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim

    def call(self, x):
        # shape=(batch_size,L,dim)
        y = []
        # 种子的循环周期为一个batch_size
        mseq = np.arange(self.L)
        for i in range(self.batch_size):
            xtmp = tf.slice(x, [i, 0, 0], [1, self.L, self.dim])
            # shape=(1,L,dim)
            xtmp = tf.reshape(xtmp, shape=(self.L, self.dim))
            # shape=(L,dim)
            np.random.seed(i)
            # 将标号按照相同的seed进行打乱
            mshuf = np.random.permutation(mseq)
            indices = tf.argsort(mshuf)
            # argsort将标号按照从小到大排序，返回值为将标号恢复正常顺序的index
            # 同理这也是被打乱bit的恢复index
            ytmp = tf.gather(xtmp, indices)
            ytmp = tf.reshape(ytmp, shape=(1, self.L, self.dim))
            # shape=(1,L,dim)
            if i == 0:
                y = ytmp
            else:
                y = tf.concat([y, ytmp], axis=0)
                # shape=(batch_size,L,dim)
        return y

# 一个batch_size进行交织
class BC_Pseudo_random_Interleaver(tf.keras.layers.Layer):
    def __init__(self, batch_size,L,dim):
        super(BC_Pseudo_random_Interleaver, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.seed = 0
    def call(self, x):
        mseq = np.arange(self.batch_size*self.L)

        xtmp = tf.reshape(x,shape=(self.batch_size*self.L,self.dim))
        np.random.seed(self.seed)
        mshuf = np.random.permutation(mseq)

        p = tf.gather(xtmp, mshuf)
        p = tf.reshape(p, shape=(self.batch_size, self.L, self.dim))

        self.seed += 1
        if self.seed == 64:
            self.seed = 0
        return p

class BC_Pseudo_random_DeInterleaver(tf.keras.layers.Layer):
    def __init__(self, batch_size,L,dim):
        super(BC_Pseudo_random_DeInterleaver, self).__init__()
        self.batch_size = batch_size
        self.L = L
        self.dim = dim
        self.seed = 0
    def call(self, x):
        mseq = np.arange(self.batch_size*self.L)

        xtmp = tf.reshape(x,shape=(self.batch_size*self.L,self.dim))
        np.random.seed(self.seed)
        mshuf = np.random.permutation(mseq)

        indices = tf.argsort(mshuf)
        y = tf.gather(xtmp, indices)
        y = tf.reshape(y, shape=(self.batch_size, self.L, self.dim))

        self.seed += 1
        if self.seed == 64:
            self.seed = 0
        return y