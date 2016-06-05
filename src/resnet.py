from keras.layers import merge, Convolution2D, Input, BatchNormalization, Flatten, Dense, Reshape, ZeroPadding2D
from keras.models import Model

class ResNet(object):
    def __init__(self,shape,depth):
        self.depth = depth
        self.shape = shape
        n = shape[1]*shape[2]
        inputs = Input(shape=(n,))
        inputs_n = Dense(n,init='uniform',activation='relu')(self.__res_block__(shape,n,inputs))
        
        for d in range(depth-1):
            inputs_n = Dense(n,init='uniform',activation='relu')(self.__res_block__(shape,n,inputs_n))

        out = Dense(10,init='uniform',activation='relu')(self.__res_block__(shape,n,inputs_n))
        self.resnet = Model(input=inputs,output=out)

    def __res_block__(self,shape,n,inputs):
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

    
