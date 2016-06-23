import pickle
import argparse
from mnist import MNIST
import numpy as np
#from resnet_func import *
from resnet import *
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10


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

    #datagen = ImageDataGenerator(featurewise_center=True)
    #datagen.fit(X_train)
    #datagen.fit(X_test)

    #initial_pl = np.array([pl for i in range(depth-1)])
    initial_pl = np.array([1-(i/depth)*(1-pl) for i in range(1,depth)])

    hist = LossAccuracyHistory()
    sample_z = SurvivalProb(depth,initial_pl,batch_size)
    
    net = ResNet((3,32,32),depth,filt_inc)
    resnet = net.get_net()
    
    sgd =SGD(lr=0.01, decay=1e-6,momentum=0.9)
    resnet.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])


    labels = to_categorical(y_train, 10)
    
    resnet.fit(X_train,labels,batch_size=batch_size,nb_epoch=epochs,callbacks=[sample_z,hist])
    #    resnet.fit_generator(datagen.flow(X_train,labels,batch_size=batch_size),samples_per_epoch=len(X_train),callbacks=[sample_z])
    labels_test = to_categorical(y_test, 10)
    resnet.test = 1
    resnet.z = initial_pl
    
    score = resnet.evaluate(X_test, labels_test, batch_size=batch_size)

    results = {}
    results[str(resnet.metrics_names)] = score
    results['losses'] = hist.losses
    results['accuracy']=hist.accuracy
    results['Z_hist']=sample_z.Z_history
    with open('results/res0', 'wb') as f:
        pickle.dump(results,f)
        
    
    
if __name__ == "__main__":
    main()
