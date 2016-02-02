
import sys
sys.stdout=open('dumpsamad','w')

print "RUNNING "

import numpy as np
import theano
import random
import os

os.chdir("/home/akhan/Documents/mnistNN/Code/")
import RandomCompression
from mnisthelper import *

os.chdir("/home/akhan/pylearn2")
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter

from pylearn2.datasets import DenseDesignMatrix

class mnistDS(DenseDesignMatrix):
        def __init__(self,X=None,y=None):
            print X.shape
            print y.shape
            super(mnistDS,self).__init__(X=X,y=y,y_labels=10)

# This contains code to read the input files
#from mnisthelper import *
# mnist Dataset of DenseDesignMatrix format (pylearn2 format)


def training(ds_train,dim_com):
    # CREATE NEURAL NEWORK 
    # Structure
    print "Creating Neural Network"
    nvisable=dim_com
    hidden_layer=mlp.RectifiedLinear(layer_name="hiddenLayer",dim=100,irange=0.05,init_bias=1.0)
    output_layer=mlp.Softmax(n_classes=10,layer_name="outputLayer",irange=0.05)
    layer_list=[hidden_layer,output_layer]
    nn=mlp.MLP(layers=layer_list,nvis=nvisable)
    
    # TRAINING
    # Intialize Trainer
    print "Creating Trainer"
    trainer=sgd.SGD(learning_rate=0.01,batch_size=100,termination_criterion=EpochCounter(30))
    trainer.setup(nn,dataset=ds_train)
    
    #Training
    print "Begin Training"
    while True:
        trainer.train(dataset=ds_train)
        nn.monitor.report_epoch()
        nn.monitor()
        if not trainer.continue_learning(nn):
            break
    print "End of Training"
    return nn

# COMPUTING ERROR
# Testing Neural Network (Misclassification Error on Test Set)


def classify(X,net,dimx):
    X.shape=(1,dimx)
    inputs =X  
    return net.fprop(theano.shared(inputs, name='inputs')).eval()

def estimateError(ds_pylearn,net,samplesize=None,noisy=0,dimx=(28*28)):
    ds=ds_pylearn.get_data()
    features=ds[0]
    labels=ds[1]
    ds_size=ds_pylearn.get_num_examples()
    if samplesize!=None:    
        if samplesize>ds_size:
            print "Warning! samplesize exceeds dataset size. Setting to maximum possible."
            if samplesize<=0:
                raise Exception("samplesize too small.")
            test_size=min(samplesize,ds_size)
    else:
            test_size=ds_size
    correct=0
    classificationCounts=[0]*10
    for i in range(test_size):
        yhat=classify(features[i],net,dimx)
        yhat=np.argmax(yhat)
        if yhat==labels[i][0]:
            correct+=1
            classificationCounts[yhat]+=1
            if ((i+1) % 10000)==0 and noisy:
                print i+1, "of",test_size
    print "Samplesize is %s"%(test_size)
    print "Classification Accuracy is %s percent." % (correct*100/test_size)
    print "Classification Counts:"
    print classificationCounts

def getcompressedData(P,tr_data_pixel_flat,ts_data_pixel_flat):
    if P==None:
        print "No Randomized Matrix Multiplied"
        return [np.asarray(tr_data_pixel_flat),np.asarray(ts_data_pixel_flat)]
    dim_com=P.shape[0]
    print "Randomized Matrix Multiplied"    
    print "Orignal Dimension    :",dim_org
    print "Compressed Dimension :",dim_com
    random.seed(1)
    D=RandomCompression.RandomMatrices.generate_diagonal(shape=(dim_org,dim_org))
    C=np.dot(P,D)
    # Apply compression C to dataset and Non-Linear tranformation (sign function)
    tr_data_mod=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),tr_data_pixel_flat))
    ts_data_mod=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),ts_data_pixel_flat))
    return [tr_data_mod,ts_data_mod]


# Variables
current_path="/home/akhan/Documents/mnistNN/1_Data/"
train_size =  60 # max 60000
test_size  = 10 # max 10000

# LOADING DATA 
[tr_data_pixel,tr_data_target]=get_labeled_data(current_path+"train-images-idx3-ubyte.gz",current_path+"train-labels-idx1-ubyte.gz",samplesize=train_size)
[ts_data_pixel,ts_data_target]=get_labeled_data(current_path+"t10k-images-idx3-ubyte.gz",current_path+"t10k-labels-idx1-ubyte.gz",samplesize=test_size)

# Flattening Features from 28x28 format to 1-D array of length 784
tr_data_pixel_flat=map(np.ravel,tr_data_pixel)
tr_data_pixel_flat=np.asarray(tr_data_pixel_flat)
ts_data_pixel_flat=map(np.ravel,ts_data_pixel)
ts_data_pixel_flat=np.asarray(ts_data_pixel_flat)

dim_org = 28*28 # Original Dimension of features


#g  : Gaussian
#d  : Discrete
#t  : topo
#h  : hankel
#nc : Non Circulant
#c  : Circulant

para_nocompress=[[28*28,None]]
para_g_nc =[[28*28],[500],[400],[300],[200],[100],[50],[10]]
para_g_c  =[[28*28],[500],[400],[300],[200],[100],[50],[10]]
para_d_nc =[[28*28],[500],[400],[300],[200],[100],[50],[10]]
para_d_c  =[[28*28],[500],[400],[300],[200],[100],[50],[10]]
para_t    =[[28*28],[500],[400],[300],[200],[100],[50],[10]]
para_h    =[[28*28],[500],[400],[300],[200],[100],[50],[10]]


for i,item in enumerate(para_g_nc):
    item.append(RandomCompression.RandomMatrices.generate_gaussian_random(shape=(item[0],dim_org)))

for i,item in enumerate(para_g_c):
    item.append(RandomCompression.RandomMatrices.generate_circulant(first_row=[random.gauss(0,1) for i in range(dim_org)],shape=(item[0],dim_org)))

for i,item in enumerate(para_d_nc):
    item.append(RandomCompression.RandomMatrices.generate_sign_random(shape=(item[0],dim_org)))

for i,item in enumerate(para_d_c):
    item.append(RandomCompression.RandomMatrices.generate_circulant(first_row=[random.sample([-1,1],1)[0] for i in range(dim_org)],shape=(item[0],dim_org)))

for i,item in enumerate(para_t):
    item.append(RandomCompression.RandomMatrices.generate_toeplitz_random(shape=(item[0],dim_org)))

for i,item in enumerate(para_h):
    item.append(RandomCompression.RandomMatrices.generate_hankel_random(shape=(item[0],dim_org)))

parameters=[("NOCOMPRESSION",para_nocompress),("GAUSS-NONCIRC",para_g_nc),("GAUSS-CIRC",para_g_c),("DISCRETE-NONCIRC",para_d_nc),("DISCRETE-CIRC",para_d_c),("DIS-TOEPLITZ",para_t),("DIS-HANKEL",para_h)]


for para in parameters:
    for item in para[1]:
        print "*"*10
        print "Type:",para[0],
        print "Compression: ",dim_org,"-->",item[0]
        [trainx,testx]=getcompressedData(item[1],tr_data_pixel_flat,ts_data_pixel_flat)
        ds_train=mnistDS(X=trainx,y=tr_data_target)
        ds_test=mnistDS(X=testx,y=ts_data_target)
        nn=training(ds_train,item[0])
        print "Begin Estimating Errors"
        estimateError(ds_pylearn=ds_train,net=nn,noisy=1,dimx=item[0])
        estimateError(ds_pylearn=ds_test,net=nn,noisy=1,dimx=item[0])

