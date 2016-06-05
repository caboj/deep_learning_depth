import argparse
from mnist import MNIST
import numpy as np

from resblock import *
from resnet import *
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD

def main():

    # load data
    mndata = MNIST('../data')
    train_data=mndata.load_training()
    test_data=mndata.load_testing()

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

    parser.set_defaults(epochs=100,samples=0,batch_size=1,depth=10)
    args = parser.parse_args()
    
    epochs=args.epochs
    samples = len(train_data[1]) if args.samples == 0 else args.samples
    batch_size=args.batch_size
    depth=args.depth
    samples_test = len(test_data[1]) if args.samples == 0 else args.samples
    
    X_train = np.array(train_data[0][:samples])
    y_train = np.array(train_data[1][:samples])
    X_test = np.array(test_data[0][:samples_test])
    y_test = np.array(test_data[1][:samples_test])


    net = ResNet((1,28,28),depth)
    resnet = net.get_net()
    sgd =SGD(lr=0.1, decay=1e-6,momentum=0.9)
    resnet.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    labels = to_categorical(y_train, 10)

    resnet.fit(X_train,labels,nb_epoch=epochs,batch_size=batch_size)

    labels_test = to_categorical(y_test, 10)
    score = resnet.evaluate(X_test, labels_test, batch_size=batch_size)
    

    
if __name__ == "__main__":
    main()
