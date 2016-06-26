from keras.layers import merge, Merge, Flatten, Convolution2D, Input, BatchNormalization, Dense, Reshape, Activation, AveragePooling2D, AveragePooling3D
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
    def __init__(self,shape,depth,filt_inc):

        self.depth = depth
        inputs = Input(shape)
        self.z = Input((depth-1,))
        self.filter_increase = filt_inc
        self.filters = 64
        self.strides = 1
        self.test = 0
        
                                                
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
            inputs_n =Switch_Layer(self.z[i],self.test)([inputs_n,prev_in])
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
    def __init__(self, zi,test,**kwargs):
        self.zi = zi
        self.test = test
        super(Switch_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.zi = np.random.randint(2)
        self.non_trainable_weights = []

    def call(self, vals, mask=None):
        
        block_out = vals[0]
        prev_out = vals[1]
        test_out = self.zi * block_out
        
        return ifelse(self.test, test_out, ifelse(self.zi,block_out,prev_out))
        
    def get_output_shape_for(self, input_shape):
        return input_shape[1]

class SurvivalProb(Callback):
    def __init__(self, depth, pl, batch_size):
        self.pl0 = pl[0]
        self.c = 8
        self.alpha = .9
        self.depth = depth
        self.pl = pl
        self.phi = [-np.log(1/pli-1) for pli in pl]
        self.delta_phi = np.zeros(len(pl))
        self.batch_size = batch_size
        self.Z_history = []
        #self.losses = []

    def on_train_begin(self,logs={}):
        self.__resample_z()

    def on_epoch_end(self,epoch,logs={}):
        py = np.log(logs.get('acc'))
        ex_deriv = lambda z_l, phi_l: z_l-(1/(np.e**-phi_l+1))
        self.delta_phi = [self.delta_phi[i]+py*ex_deriv(self.model.z[i],self.phi[i]) for i in range(len(self.pl))]
        self.__resample_z()
        #self.losses.append(logs.get('loss'))
        
    def on_batch_end(self,batch,logs={}):
        self.__update_pl(logs.get('loss'))
        self.__resample_z()

    def __resample_z(self):
        self.model.z = np.array([binomial(1,pli) for pli in self.pl])
        self.Z_history.append(self.model.z)
        #print(' resampled z ')
        #print(self.pl)
        #print(self.model.z)

    def __update_pl(self,loss):
        cb = np.log(loss)
        self.c = self.alpha*self.c+(1-self.alpha)*cb
        li = np.log(loss) - self.c
        kl_const = lambda phi_l: np.e**phi_l/(np.e**phi_l+1)**2
        kl_deriv = lambda phi_l: kl_const(phi_l)*(np.log(self.pl0*(np.e**-phi_l+1))-np.log((1-self.pl0)*np.e**-phi_l+1))
        s = lambda x: 1/(1+np.exp(x))
        self.phi = [self.phi[i] + li*self.delta_phi[i]/self.batch_size+kl_deriv(self.phi[i]) for i in range(len(self.phi))]
        self.delta_phi=np.zeros(len(self.pl))
        pl = [s(self.phi[i]) for i in range(len(self.pl))]
        self.pl=np.array(pl)
        
        

class LossAccuracyHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('val_acc'))
