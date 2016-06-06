from keras.layers import merge, Convolution2D, Input, BatchNormalization, Flatten, Dense, Reshape, ZeroPadding2D
from keras.models import Model
from keras.callbacks import Callback
from numpy.random import binomial
import numpy as np

class ResNet(object):
    def __init__(self,shape,depth,pl):

        self.depth = depth
        self.pl = pl
        n = shape[1]*shape[2]
        inputs = Input(shape=(n,))
        self.z = np.arange(depth-1)#Input(shape=(depth-1,))
        
        inputs_n = Dense(n,init='uniform',activation='relu')(self.__res_block__(shape,n,inputs,1))
        for zi in self.z:
            inputs_n = Dense(n,init='uniform',activation='relu')(self.__res_block__(shape,n,inputs_n,zi))

        out = Dense(10,init='uniform',activation='relu')(inputs_n)#self.__res_block__(shape,n,inputs_n))
        self.resnet = Model(input=inputs,output=out)

    def __res_block__(self,shape,n,inputs,zi):

        if zi == 0:
            return inputs 
    
        x = Reshape(shape,input_shape=(n,))(inputs)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = Convolution2D(1,3,3,init='uniform',border_mode='valid')(x)
        x = BatchNormalization(mode=1)(x)
        x = Flatten()(x)
        x = Dense(n,init='uniform',activation='relu')(x)
        x = Reshape(shape)(x)
        x = ZeroPadding2D(padding=(1,1))(x)
        x = Convolution2D(1,3,3,init='uniform',border_mode='valid')(x)
        x = BatchNormalization(mode=1)(x)
        x = Flatten()(x)

        block_out = merge([inputs,x],mode='sum')

        return block_out
        
    def get_net(self):
        return self.resnet

class SurvivalProb(Callback):
    def __init__(self, depth, pl):
        self.depth = depth
        self.pl = pl
        
    def on_train_begin(self,logs={}):
        self.__resample_z()

    def on_batch_end(self,batch,logs={}):
        self.__resample_z()

    def __resample_z(self):
        self.model.z = np.array([binomial(1,1-(d/self.depth)*(1-self.pl)) for d in range(1,self.depth)])
    
