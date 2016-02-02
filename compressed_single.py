# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 17:12:18 2015

 

References:
    www.arngarden.com
    www.deeplearning.net/software/pylearn2
    
Update Log:
    * Neural Network working with accuracy 95% and 94% on Training and Test set respectively
    * Implemented own Data Set libarary compatable with pylearn2 (DenseDesignMatrix) to allow manipulation of data points
    * Compressing Data points before 
    * Replacing Linear Rectifier by     

"""
import os

os.chdir("/home/akhan/pylearn2")

from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter

os.chdir("/home/akhan/Documents/mnistNN/Code/")

# This contains code to read the input files
from mnisthelper import *
# mnist Dataset of DenseDesignMatrix format (pylearn2 format)
from mnistDS import mnistDS
import numpy as np

# Variables
current_path="/home/akhan/Documents/mnistNN/1_Data/"
train_size =  600 # max 60000
test_size  = 100 # max 10000

# LOADING DATA 
[tr_data_pixel,tr_data_target]=get_labeled_data(current_path+"train-images-idx3-ubyte.gz",current_path+"train-labels-idx1-ubyte.gz",samplesize=train_size)
[ts_data_pixel,ts_data_target]=get_labeled_data(current_path+"t10k-images-idx3-ubyte.gz",current_path+"t10k-labels-idx1-ubyte.gz",samplesize=test_size)

# Flattening Features from 28x28 format to 1-D array of length 784
tr_data_pixel_flat=map(np.ravel,tr_data_pixel)
tr_data_pixel_flat=np.asarray(tr_data_pixel_flat)
ts_data_pixel_flat=map(np.ravel,ts_data_pixel)
ts_data_pixel_flat=np.asarray(ts_data_pixel_flat)
dim_com=28*28

# COMPRESSION
import RandomCompression
import random
dim_org = 28*28 # Original Dimension of features
#dim_com = 200   # Compressed Dimension of features
print "Orignal Dimension    :",dim_org
print "Compressed Dimension :",dim_com

## ---- Replicating Paper's Results for Circulant Matrix ------------------------
#random.seed(1)
#rand_gaussian_row=[random.gauss(0,1) for i in range(dim_org)]
#D=RandomCompression.RandomMatrices.generate_diagonal(shape=(dim_org,dim_org))
#P=RandomCompression.RandomMatrices.generate_circulant(first_row=rand_gaussian_row,shape=(dim_com,dim_org))
#C=np.dot(P,D)
## Apply compression C to dataset and Non-Linear tranformation (sign function)
#tr_data_pixel_flat=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),tr_data_pixel_flat))
#ts_data_pixel_flat=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),ts_data_pixel_flat))
##------------------------------------------------------------------------------

# ---- Binary Circulant Matrix ------------------------
random.seed(1)
rand_bin_row=row=[random.sample([-1,1],1)[0] for i in range(dim_org)]
D=RandomCompression.RandomMatrices.generate_diagonal(shape=(dim_org,dim_org))
P=RandomCompression.RandomMatrices.generate_circulant(first_row=rand_bin_row,shape=(dim_com,dim_org))
C=np.dot(P,D)
# Apply compression C to dataset and Non-Linear tranformation (sign function)
tr_data_pixel_flat=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),tr_data_pixel_flat))
ts_data_pixel_flat=np.asarray(map(lambda x: np.sign(np.dot(a=C,b=x)),ts_data_pixel_flat))
#------------------------------------------------------------------------------

# Creating Dataset of DenseDesignMatrix format
ds_train=mnistDS(X=tr_data_pixel_flat,y=tr_data_target)
ds_test=mnistDS(X=ts_data_pixel_flat,y=ts_data_target)

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

# COMPUTING ERROR
# Testing Neural Network (Misclassification Error on Test Set)
import theano
import numpy as np
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
            if ((i+1) % 1000)==0 and noisy:
                print i+1, "of",test_size
    print "Samplesize is %s"%(test_size)
    print "Classification Accuracy is %s percent." % (correct*100/test_size)
    print "Classification Counts:"
    print classificationCounts

print "Begin Estimating Errors"
estimateError(ds_pylearn=ds_train,net=nn,noisy=1,dimx=nvisable)
estimateError(ds_pylearn=ds_test,net=nn,noisy=1,dimx=nvisable)

