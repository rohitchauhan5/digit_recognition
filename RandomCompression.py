# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 18:21:55 2015


"""
import numpy as np
import random
class RandomMatrices(object):
    def __init__():
        pass
    def dotinbinary(x,y):
        binx=list(bin(x))
        biny=list(bin(y))
        binx=binx[2:]
        biny=biny[2:]
        binx=list(reversed(binx))
        biny=list(reversed(biny))
        return sum([int(binx[i])*int(biny[i]) for i in range(min(len(binx),len(biny)))])
    
    @staticmethod # Returns a Circulant Matrix using the first row
    def generate_circulant(first_row=None,shape=None):
        if shape==None:
            raise Exception('No shape was specified!')
        if first_row==None:
            raise Exception('No first_row specified')        
        if shape[1]!=len(first_row):
            raise Exception('first_row and shape are inconsistant!')        
        circulant=first_row        
        for i in range(shape[0]-1):
            shiftedrow=np.roll(first_row,shift=(i+1))
            circulant=np.vstack([circulant,shiftedrow])
        return circulant
        
    @staticmethod
    def generate_gaussian_random(shape=None,q=1,norm_mean=0,norm_sd=1):
        if shape==None:
            raise Exception('No shape was specified!')
        if q<0 or q>1:
            raise Exception('Probabilty q is not between 0 and 1')
        print "q is set to ",q
        normalMat = np.random.normal(loc=norm_mean,scale=norm_sd,size=shape)
        uniformMat= np.random.uniform(low=0,high=1,size=shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if uniformMat[i][j]<1-q:
                    normalMat[i][j]=0
        return normalMat
    @staticmethod
    def generate_sign_random(shape=None):
        if shape==None:
            raise Exception('No shape was specified!')
        row=[random.sample([-1,1],1)[0] for i in range(shape[1])]
        mat=row
        for i in range(shape[0]-1):
            row=[random.sample([-1,1],1)[0] for i in range(shape[1])]
            mat=np.vstack([mat,row])
        return mat
    @staticmethod
    def generate_toeplitz_random(shape=None):
        if shape==None:
            raise Exception('No shape was specified!')
        row=[random.sample([-1,1,3,4,5,6,7,8,9,8,6,7,9],1)[0] for i in range(shape[1])]
        mat=row
        for i in range(shape[0]-1):
            row=[random.sample([-1,1,3,4,5,6,7,8,9,8,6,7,9],1)[0] for i in range(shape[1])]
            mat=np.vstack([mat,row])
        for i in range(shape[1]):
            k=i
            for j in range(shape[0]):
                if k==shape[1] or j==shape[0]:
                    break
                #print (0,i),"=",(j,k)                
                mat[j][k]=mat[0][i]
                k+=1
        for i in range(shape[0]):
            k=i
            for j in range(shape[1]):
                if k==shape[0] or j==shape[1]:
                    break
                #print (i,0),"=",(k,j)                
                mat[k][j]=mat[i][0]
                k+=1
        return mat

    @staticmethod
    def generate_hankel_random(shape=None):
        mat=RandomMatrices.generate_toeplitz_random(shape=(shape[1],shape[0]))
        return np.rot90(mat)
 
    @staticmethod
    def generate_walsh_hadamard(shape=None):
        if shape==None or (type(shape) is not tuple) or len(shape)!=2:
            raise Exception('No shape was specified or was incorrectly specified!')
        d=shape[0]
        print "Matrix dim set to %s x %s"%(d,d)
        wh=np.ones(shape=(d,d))
        for i in range(d):
            for j in range(d):
                wh[i][j]=(1/np.sqrt(d))*(-1)**(dotinbinary(i,j) % 2)
        return wh
    @staticmethod
    def generate_diagonal(shape=None):
        if shape==None or (type(shape) is not tuple) or len(shape)!=2:
            raise Exception('No shape was specified or was incorrectly specified!')
        d=shape[0]
        print "Matrix dim set to %s x %s"%(d,d)
        arr=[random.sample([-1,1],1)[0] for x in range(d)]
        return np.diag(np.asarray(arr))           

if __name__=="__main__":
    pass    
    #RandomCompression.generate_normal_random(shape=(2,3))
        
