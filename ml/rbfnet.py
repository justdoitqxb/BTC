#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import time
import numpy as np
import matplotlib.pyplot as plt
from util.generator import Generator

class rbf(object):
    def __init__(self, num_centers, regression = True):
        self.num_centers = num_centers
        self.kxi = 0.3
        
    def _centers(self, inputsize,x, type = 'rand'):
        if type == 'rand':
            return [np.random.uniform(-10,10,inputsize) for i in xrange(self.num_centers)]
        elif type == 'randpick':
            num_samples = x.shape[0]
            randindex = [i for i in xrange(num_samples)]            
            np.random.shuffle(randindex)
            print x[randindex[0:self.num_centers],:]
            return x[randindex[0:self.num_centers],:]
        else:
            raise NameError('No the type usable!')
    
    def _weight(self, outputsize):
        return np.random.random((outputsize, self.num_centers))
    
    def _basisfunc(self, x):
        xx = np.tile(x,(self.num_centers,1))
        return np.exp(-np.sum(np.power((xx - self.centers),2),1)/2.0/self.kxi**2)
    
    def _normalize(self,x):
        n = x.shape[1]
        x_sum = np.sum(x,1)
        x_tile = np.tile(x_sum,(1,n))
        return x / x_tile
    
    def _train(self, x, y):
        start_time = time.time()
        inputsize = x.shape[1]
        num_samples = x.shape[0]
        #x = self._normalize(x)
        self.centers = self._centers(inputsize,x,'randpick')
        G = np.mat(np.zeros((self.num_centers,num_samples),np.float))
        for i in xrange(num_samples):
            G[:,i] = self._basisfunc(x[i,:])
        self.W = np.dot(np.linalg.pinv(G.T),y)  
        end_time = time.time()
        print 'Congratulations, training complete! Took %fs!' % (end_time - start_time)   
        
    def _pridict(self,x):
        inputsize = x.shape[1]
        num_samples = x.shape[0]
        G = np.mat(np.zeros((self.num_centers,num_samples),np.float))
        for i in xrange(num_samples):
            G[:,i] = self._basisfunc(x[i,:])
        return np.dot(G.T, self.W)
    
if __name__ == '__main__':
    
    dataSet = []
    labels = []
    gt = Generator()
    
    data = gt.load_points("../data/reg_points.txt")  

    data = np.mat(data)

    dataSet = data[:,0:1]
    labels = data[:,1:2]
    train_x = dataSet[0:80,:]
    train_y = labels[0:80,:]
    test_x = dataSet[80:100,:]
    test_y = labels[80:100,:]
    
    rbf = rbf(50)    
    rbf._train(train_x, train_y)
    print rbf.W
    print rbf.centers
    print test_y.T
    print rbf._pridict(test_x).T
    
    
    
