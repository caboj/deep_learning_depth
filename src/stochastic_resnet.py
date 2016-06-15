import argparse
from mnist import MNIST
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

from keras.layers import merge, Merge, Flatten, Convolution3D, Convolution2D, Input, BatchNormalization, Dense, Reshape, Activation, AveragePooling2D, AveragePooling3D
from keras.models import Model, Sequential
from keras.callbacks import Callback
from keras import backend as K
import theano.tensor as T
from theano.ifelse import ifelse
from numpy.random import binomial
import numpy as np
from keras.engine.topology import Layer
def main():

    # load data
    #mndata = MNIST('../data')
    #train_data=mndata.load_training()
    #test_data=mndata.load_testing()

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # parse aguments
    parser = argparse.ArgumentParser(description='train simple resnet')
    parser.add_argument('-e', metavar='epochs', dest="epochs",type=int,
                        help='number of training epochs')
    parser.add_argument('-b', metavar='batch_size', dest="batch_size",type=int,
                        help='batch_size to train with')
    parser.add_argument('-s', metavar='samples', dest="samples",type=int,
                        help='number of training samples to use')
    parser.add_argument('-d', metavar='depth', dest="depth",type=int,
                        help='number of reblocks to use')
    parser.add_argument('-p', metavar="survival_p", dest="survival_p", type=float, 
                        help='set this to use stochastic resnet')
    parser.add_argument('-f', metavar="filter_increase", dest="filt_inc", nargs='+', 
                        help='specify at which resblocks the filters should double')

    parser.set_defaults(epochs=100,samples=0,batch_size=20,depth=17,survival_p=.5,filt_inc=[3,7,13])
    args = parser.parse_args()
    
    epochs=args.epochs
    samples = len(train_data[1]) if args.samples == 0 else args.samples
    batch_size=args.batch_size
    depth=args.depth
    samples_test = len(test_data[1]) if args.samples == 0 else args.samples
    pl=args.survival_p
    filt_inc = args.filt_inc

    X_train = np.array(X_train[:samples])
    y_train = np.array(y_train[:samples])
    X_test = np.array(X_test[:samples_test])
    y_test = np.array(y_test[:samples_test])

    z = np.array([binomial(1,1-(d/depth)*(1-pl)) for d in range(1,depth)]) #.reshape(1,16)
    
    resnet = get_resnet((3,32,32), depth, filt_inc,z)
    sgd =SGD(lr=0.001, decay=1e-4,momentum=0.9)
    resnet.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    labels = to_categorical(y_train, 10)
    #sample_z = SurvivalProb(depth,pl)
    
    #Z = np.array([z for e in range(batch_size)])

    resnet.fit([X_train],labels,nb_epoch=epochs,batch_size=batch_size)
    
    labels_test = to_categorical(y_test, 10)
    #resnet.z = np.ones(depth)
    score = resnet.evaluate(X_test, labels_test, batch_size=batch_size)

    print(score)


def get_resnet(shape, depth, filter_increase, initial_z):
    inputs = Input(shape)
    z = initial_z #Input(shape=(depth-1,))
    filters = 64
    strides = 1
    old_shape = shape

    inputs_n = Activation('relu')(
        BatchNormalization(mode=0,axis=1)(
            Convolution2D(64,7,7,border_mode='same')(inputs)))

    for i in range(depth-1):
        if i in filter_increase:
            strides = 2 
            filters = int(filters*2)
            shape = (filters,int(shape[1]/2),int(shape[2]/2))
        else:
            old_shape = shape
            strides=1
        
        prev_in = Convolution2D(filters,1,1,border_mode='same',subsample=(strides,strides))(inputs_n)
        inputs_n = res_block(filters, strides, inputs_n)
        
        inputs_n = Switch_Layer(z[i])([inputs_n,prev_in])
    out = AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs_n)
    out = Dense(10,activation='softmax')(Flatten()(out))

    return Model([inputs],out)

def res_block(filters, strides, inputs):

    x = Convolution2D(filters,3,3,border_mode='same',subsample=(strides,strides))(inputs)
    x = BatchNormalization(mode=0,axis=1)(x)
    x = Activation('relu')(x)
    x = Convolution2D(filters,3,3,border_mode='same')(x)
    x = BatchNormalization(mode=0,axis=1)(x)

    i = Convolution2D(filters,1,1,border_mode='same',subsample=(strides,strides))(inputs)
    
    block = merge([i,x],mode='sum')
    block = Activation('relu')(block)
    
    return block

class Switch_Layer(Layer):
    def __init__(self, zi, **kwargs):
        #self.orig = orig
        self.zi = zi
        super(Switch_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.trainable_weights = []

    def call(self, vals, mask=None):
        x=vals[0]
        x0 = vals[1]
        return ifelse(K.equal(0,self.zi),x0,x)#self.orig,x)
        
    def get_output_shape_for(self, input_shape):
        return input_shape[0]
    
if __name__ == "__main__":
    main()
