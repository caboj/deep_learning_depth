from keras.layers import merge, Merge, Flatten, Convolution3D, Convolution2D, Input, BatchNormalization, Dense, Reshape, Activation, AveragePooling2D, AveragePooling3D
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras import backend as K
import theano.tensor as T
from theano.ifelse import ifelse
from numpy.random import binomial
import numpy as np
import theano
from keras.engine.topology import Layer

class ResNet(object):
    def __init__(self,shape,depth,pl,filt_inc):

        self.depth = depth
        inputs = Input(shape)
        self.pl = pl
        self.z = np.array([binomial(1,1-(d/self.depth)*(1-self.pl)) for d in range(1,self.depth)])
        self.filter_increase = filt_inc
        self.filters = 64
        self.strides = 1
                                                
        inputs_n = Activation('relu')(
            BatchNormalization(mode=0,axis=1)(
                Convolution2D(64,7,7,border_mode='same')(inputs)))

        for i in range(depth-1):
            if i in self.filter_increase:
                self.strides = 2 
                self.filters = int(self.filters*2)
                shape = (self.filters,int(shape[1]/2),int(shape[2]/2))
            else:
                self.strides=1

            prev_in = Convolution2D(self.filters,1,1,border_mode='same',subsample=(self.strides,self.strides))(inputs_n)
            inputs_n = self.__res_block__(inputs_n)
            inputs_n = Switch_Layer(self.z[i])([inputs_n,prev_in])

        out = AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs_n)
        out = Dense(10,activation='softmax')(Flatten()(out))

        self.resnet = Model(input=inputs,output=out)

    def __res_block__(self,inputs):

        i = Convolution2D(self.filters,3,3,border_mode='same',subsample=(self.strides,self.strides))(inputs)
        x = BatchNormalization(mode=0,axis=1)(i)
        x = Activation('relu')(x)
        x = Convolution2D(self.filters,3,3,border_mode='same')(x)
        x = BatchNormalization(mode=0,axis=1)(x)
        
        block = merge([i,x],mode='sum')
        block = Activation('relu')(block)

        return block
        
    def get_net(self):
        return self.resnet

class Switch_Layer(Layer):
    def __init__(self, zi, **kwargs):
        self.zi = zi
        super(Switch_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, vals, mask=None):
        x=vals[0]
        x0 = vals[1]
        return ifelse(K.equal(0,self.zi),x0,x)
        
    def get_output_shape_for(self, input_shape):
        return input_shape[0]

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
    
