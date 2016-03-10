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
from kernels import *
from actionfunc import *
import numpy as np
from util.generator import Generator
 
class ELM(object):
     
    def __init__(self, hidden_size, action_func, initial_method,C = 10**3 ,numcase = None, kerne_type= None, classifier = True, regression = False):
        self.hidden_size = hidden_size        
        self.initial_method = initial_method
        self.C = C
        self.classifier = classifier
        self.regression = regression
        self.kernel_type = kerne_type
        if classifier:
            self.numcase = numcase
            
        if action_func == 'sigmoid':
            self.action_func = sigmoid()
        elif action_func  == 'sine':
            self.action_func = sine()
        elif action_func == 'hardlim':
            self.action_func = hardlim()
        elif action_func == 'rbfnet':
            self.action_func = rbfnet()
        else:
            raise ValueError('unknow active function')
        
        if kerne_type == 'linear':
            self.kernel = linear_kernel()
        elif kerne_type  == 'polynomial':
            self.kernel = polynomial_kernel(1.0,3.0)
        elif kerne_type == 'rbf':
            self.kernel = rbf_kernel(1.0)
        elif kerne_type == 'sigmoid':
            self.kernel = sigmoid_kernel(1.0, 0.5)
        elif kerne_type == 'wavelet':
            self.kernel = wavelet_kernel(1.0,2.0,0.5)
        elif kerne_type == None:
            pass
        else:
            raise NameError('unknow kernel methond')        
        
    def init_weight(self):        
        if self.initial_method == 'random':
            #Initialize parameters randomly based on layer sizes
            #choose weights uniformly from the interval[-r,r]
            r = np.sqrt(6) / np.sqrt(self.hidden_size + self.inputsize + 1) 
            weight = np.random.rand(self.hidden_size, self.inputsize) * 2 * r - r
            bias = np.random.rand(self.hidden_size,1)  
        elif self.initial_method == 'rand':
            weight = np.random.rand(self.hidden_size, self.inputsize) * 2  - 1
            bias = np.random.rand(self.hidden_size,1)  
            #bias = np.random.rand(self.hidden_size,1) / 3.0 + 1 / 11.0  for image segment  
            #bias = np.random.rand(self.hidden_size,1) / 20.0 + 1 / 60.0  for  DNA    
          
        return np.mat(weight),np.mat(bias)
    
    def _normalize(self,x):
        n = x.shape[0]
        x_sum = np.sum(x,0)
        x_tile = np.tile(x_sum,(n,1))
        return x/x_tile        
    
    # calculate kernel matrix given train set and kernel type  
    def _kernel_matrix(self, train_x,sample_x):  
        train_num = train_x.shape[0]
        sample_num = sample_x.shape[0]  
        kernel_matrix = np.mat(np.zeros((train_num, sample_num),np.float))  
        for i in xrange(sample_num):  
            kernel_matrix[:,i] = self.kernel(train_x, sample_x[i,:])  
        return kernel_matrix 
    
    def _train(self, x ,y):
         # calculate training time  
        start_time = time.time() 
        sample_num = x.shape[1]
        self.inputsize = x.shape[0]
        if self.classifier:
            y_target = np.zeros((self.numcase, sample_num),np.float)
            for i in xrange(sample_num):
                y_target[int(y[0,i]),i] = 1
            y_target = np.mat(y_target)
        elif self.regression:
            y_target = y       
        
        if self.kernel_type == None:    
            w1, bias = self.init_weight()
            #x = self._normalize(x)
            if self.action_func == 'rbfnet':
                V = np.zeros((self.hidden_size,sample_num),np.float)
                for j in xrange(self.hidden_size):
                    ww = w1[j,:]
                    ww_tmp = np.tile(ww,(sample_num,1))
                    V[:,j] = -np.sum(np.power(x.T - ww_tmp,2),1)
                H = self.action_func(np.multiply(V.T,np.tile(bias,(1,sample_num))))
            else:
                H = self.action_func(np.dot(w1, x) + np.tile(bias, (1,sample_num))) 
            beta  = np.dot(np.linalg.pinv(H.T),y_target.T)
            self.w1 = w1
            self.bias = bias
        else:
            omega = self._kernel_matrix(x.T,x.T)            
            beta = np.linalg.solve(omega + np.mat(np.eye(sample_num))/self.C,y_target.T)    
            self.train_x = x  
        end_time = time.time()
        print 'Congratulations, training complete! Took %fs!' % (end_time - start_time)  
        self.beta = beta 
                
    def _pridict(self, x):
        sample_num = x.shape[1]
        #x = self._normalize(x)
        if self.kernel_type == None:
            H = self.action_func(np.dot(self.w1, x) + np.tile(self.bias, (1,sample_num)))
            target = np.dot(H.T,self.beta).T
        else:
            omega = self._kernel_matrix(self.train_x.T, x.T)    
            target = np.dot(self.beta.T,omega)
        
        if self.regression:
            return target
        elif self.classifier:
            map_target = np.zeros((1,sample_num),np.float)
            max = target.max(0)
            for i in xrange(sample_num):
                for j in xrange(self.numcase):
                    if(max[0,i]==target[j,i]):
                        map_target[0,i] = j
                        break
            return map_target
    
if __name__ == '__main__':
    dataSet = []
    labels = []
    gt = Generator()
    
    data = gt.load_points("../data/reg_points.txt")  

    data = np.mat(data)

    dataSet = data[:,0:1]
    labels = data[:,1:2]
    train_x = dataSet[0:80,:].T
    train_y = labels[0:80,:].T
    test_x = dataSet[80:100,:].T
    test_y = labels[80:100,:].T
    elm = ELM(50,'rbfnet', 'rand', numcase=2, kerne_type= 'rbf',classifier=False, regression=True)
    elm._train(train_x, train_y)
    #print elm.beta
    print elm._pridict(test_x)
    print test_y
    
    