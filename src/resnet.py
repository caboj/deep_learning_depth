from keras.layers import merge, Flatten, Convolution3D, Convolution2D, Input, BatchNormalization, Dense, Reshape, Activation, AveragePooling2D, AveragePooling3D
from keras.models import Model
from keras.callbacks import Callback
from numpy.random import binomial
import numpy as np

class ResNet(object):
    def __init__(self,shape,depth,pl,filt_inc):

        self.depth = depth
        self.pl = pl
        n = shape[1]*shape[2]
        inputs = Input(shape)
        self.z = np.arange(depth-1)#Input(shape=(depth-1,))
        self.filter_increase = filt_inc
        self.filters = 64
        self.strides = 1
        
        inputs_n = Activation('relu')(BatchNormalization(mode=1)(Convolution2D(64,7,7,border_mode='same')(inputs)))#(Reshape(shape)(inputs))))
        dim_decrease = 1
        for i in range(len(self.z)):
            if i in self.filter_increase:
                dim_decrease += 1
                self.strides = 2
                self.filters = int(self.filters*2)
            else:
                self.strides=1
            inputs_n = Activation('relu')(self.__res_block__(shape,n,inputs_n,self.z[i]))


        int(shape[2]/2**dim_decrease)
        out = AveragePooling2D(pool_size=(int(shape[1]/2**dim_decrease),int(shape[2]/2**dim_decrease)))(inputs_n)
        out = Dense(10,activation='softmax')(Flatten()(out))
        self.resnet = Model(input=inputs,output=out)

    def __res_block__(self,shape,n,inputs,zi):

        if zi == 0:
            return inputs 
    
        x = Convolution2D(self.filters,3,3,border_mode='same',subsample=(self.strides,self.strides))(inputs)
        x = BatchNormalization(mode=1)(x)
        x = Activation('relu')(x)
        x = Convolution2D(self.filters,3,3,border_mode='same')(x)
        x = BatchNormalization(mode=1)(x)
        
        i = Convolution2D(self.filters,1,1,border_mode='same',subsample=(self.strides,self.strides))(inputs)
        block_out = merge([i,x],mode='sum')

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
        if self.pl == 1:
            self.model.z = np.ones(depth)
        else:
            self.model.z = np.array([binomial(1,1-(d/self.depth)*(1-self.pl)) for d in range(1,self.depth)])
    
