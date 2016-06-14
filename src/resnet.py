from keras.layers import merge, Merge, Flatten, Convolution3D, Convolution2D, Input, BatchNormalization, Dense, Reshape, Activation, AveragePooling2D, AveragePooling3D
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras import backend as K
import theano.tensor as T
from theano.ifelse import ifelse
from numpy.random import binomial
import numpy as np


class ResNet(object):
    def __init__(self,shape,depth,pl,filt_inc):

        self.depth = depth
        self.pl = pl
        n = shape[1]*shape[2]
        inputs = Input(shape)
        self.z = Input(shape=(depth-1,))
        self.filter_increase = filt_inc
        self.filters = 64
        self.strides = 1
        old_shape = shape
                
        #net = Sequential([Convolution2D(64,7,7,border_mode='same',input_shape=shape),
        #                  BatchNormalization(mode=0,axis=1),
        #                  Activation('relu')])
                                        
        inputs_n = Activation('relu')(BatchNormalization(mode=0,axis=1)(Convolution2D(64,7,7,border_mode='same')(inputs)))

        for i in range(depth-1):
            if i in self.filter_increase:
                self.strides = 2 
                self.filters = int(self.filters*2)
                shape = (self.filters,int(shape[1]/2),int(shape[2]/2))
            else:
                old_shape = shape
                self.strides=1
            #net.add(self.__res_block__(shape,old_shape,inputs,self.z[i]))
            #inputs_n = self.__res_block__(shape,old_shape,inputs_n,self.z[i])
            conv = Convolution2D(self.filters,1,1,border_mode='same',subsample=(self.strides,self.strides))
            conv.build(shape)

            # --- comment l.46  and uncomment l.47 to see error introduced by K.switch
            inputs_n = K.switch(T.eq(1,self.z[i]),
                                conv.call(inputs_n),
                                conv.call(inputs_n))
                                #Convolution2D(self.filters,1,1,border_mode='same',subsample=(self.strides,self.strides))(inputs_n))
        #net.add(AveragePooling2D(pool_size=(shape[1],shape[2])))
        #net.add(Flatten())
        #net.add(Dense(10,activation='softmax'))
        #out = AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs_n)
        #out = Dense(10,activation='softmax')(Flatten()(out))

        #self.resnet = net
        self.resnet = Model(input=inputs,output=inputs_n)

    def __res_block__(self,shape,old_shape,inputs,zi):

        '''
        res_block = Sequential([
            Convolution2D(self.filters,3,3,border_mode='same',subsample=(self.strides,self.strides),input_shape=old_shape),
            BatchNormalization(mode=0,axis=1),
            Activation('relu'),
            Convolution2D(self.filters,3,3,border_mode='same'),
            BatchNormalization(mode=0,axis=1)])

        pass_block = Sequential([Convolution2D(self.filters,1,1,border_mode='same',subsample=(self.strides,self.strides),input_shape=old_shape)])

        block = Merge([res_block,pass_block],mode='sum')

        block_out = Sequential([ block,Activation('relu')])
            
        '''
        i = Convolution2D(self.filters,3,3,border_mode='same',subsample=(self.strides,self.strides))(inputs)
        x = BatchNormalization(mode=0,axis=1)(i)
        x = Activation('relu')(x)
        x = Convolution2D(self.filters,3,3,border_mode='same')(x)
        x = BatchNormalization(mode=0,axis=1)(x)
        
        block = merge([i,x],mode='sum')
        block = Activation('relu')(block)

        #res_block = Model(input=inputs,output=block)
        
        return block
        
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
    
