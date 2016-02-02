# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 13:44:22 2015

@author: akhan
"""
from pylearn2.datasets import DenseDesignMatrix

class mnistDS(DenseDesignMatrix):
        def __init__(self,X=None,y=None):
            print X.shape
            print y.shape
            super(mnistDS,self).__init__(X=X,y=y,y_labels=10)
            
            
