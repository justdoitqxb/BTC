#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
from util.generator import *
import numpy as np
from cvxopt.solvers import qp
from cvxopt.base import matrix
import matplotlib.pyplot as plt

class SVM(object):
    def __init__(self,  C=1.0, epsilon=10**-5, kernel_option = ('linear',1.0), with_slack=False):
        
        self.kernel_opt = kernel_option
        self.epsilon = epsilon
        self.C = C
        self.with_slack = with_slack
        
    # calculate kernel value  
    def kernel(self,matrix_x, sample_x, kernel_option):  
        kernelType = kernel_option[0]  
        numSamples = matrix_x.shape[0]  
        kernelValue = np.mat(np.zeros((numSamples, 1)))  
      
        if kernelType == 'linear':  
            kernelValue = matrix_x * sample_x.T + 1
        elif kernelType == 'polynomial':
            R = kernel_option[1]
            degree = kernel_option[2]
            kernelValue = np.power(matrix_x * sample_x.T + R,degree)
        elif kernelType == 'rbf':  
            sigma = kernel_option[1]  
            if sigma == 0:  
                sigma = 1.0  
            for i in xrange(numSamples):  
                diff = matrix_x[i, :] - sample_x  
                kernelValue[i] = np.exp(diff * diff.T / (-2.0 * sigma**2)) 
        elif kernelType == 'sigmoid':
            k = kernel_option[1]
            delta = kernel_option[2]        
            kernelValue = np.tanh(k * matrix_x * sample_x.T - delta)
        else:  
            raise ValueError('Kernel '+kernelType+' not available') 
        return kernelValue  
  
    def train(self, train_x, labels):
        return self.solve_optimization(train_x, labels)

    def indicator(self, sample):
        return sum([sv[0]*sv[2]*self.kernel(sv[1],np.mat(sample),self.kernel_opt)
                    for sv in self.support_vector])

    def predict(self, sample):
        if self.indicator(sample)>0:
            return 1
        else:
            return -1 
    
    def build_Q(self, train_x, labels):
        num_samples = train_x.shape[0]  
        Q_matrix = np.mat(np.zeros((num_samples, num_samples)))  
        for i in xrange(num_samples):  
            Q_matrix[:, i] = np.multiply(labels,self.kernel(train_x, train_x[i, :], self.kernel_opt)) * labels[i]
        return Q_matrix 
    
    def solve_optimization(self, train_x, labels):
        Q = self.build_Q(train_x, labels)
        p = [-1.0] * train_x.shape[0]

        if self.with_slack:
            h = [0.0] * train_x.shape[0] + [self.C] * train_x.shape[0]
            G = np.concatenate((np.identity(train_x.shape[0]) * -1,
                                np.identity(train_x.shape[0])))
        else:
            h = [0.0] * train_x.shape[0]
            G = np.identity(train_x.shape[0]) * -1

        optimized = qp(matrix(Q), matrix(p), matrix(G), matrix(h))
        if optimized['status'] == 'optimal':
            alphas = list(optimized['x'])

            self.support_vector = [(alpha, train_x[i,:],labels[i])
                                   for i,alpha in enumerate(alphas)
                                   if alpha>self.epsilon]
            return True
        else:
            print "No valid separating hyperplane found"
            print "Find a best hyperplane for the mixture data..."
            h = [0.0] * train_x.shape[0] + [self.C] * train_x.shape[0]
            G = np.concatenate((np.identity(train_x.shape[0]) * -1,
                                np.identity(train_x.shape[0])))
            optimized = qp(matrix(Q), matrix(p), matrix(G), matrix(h))
            alphas = list(optimized['x'])

            self.support_vector = [(alpha, train_x[i,:],labels[i])
                                   for i,alpha in enumerate(alphas)
                                   if alpha>self.epsilon]
            return False

    # testing your trained svm model given test set  
    def testsvm(self, test_x, test_y):  
        #test_x = np.mat(test_x)  
        #test_y = np.mat(test_y)  
        number = test_x.shape[0]  
        matchCount = 0  
        for i in xrange(number):   
            predict_value = self.predict(test_x[i,:]) 
            if np.sign(predict_value) == np.sign(test_y[i]):  
                matchCount += 1  
        accuracy = float(matchCount) / number  
        return accuracy  
    
    def print_data(self, points, labels):
        number = points.shape[0]
        for i in range(number):
            if labels[i] == 1:
                plt.plot(points[i,0],
                           points[i,1],
                           'bo')
            else:
                plt.plot(points[i,0],
                           points[i,1],
                           'ro')
        
    def print_boundaries(self):
        xrange = np.arange(-5,8,0.2)
        yrange = np.arange(-5,8,0.2)
        grid = matrix([[np.array(self.indicator((x,y)))[0][0] for y in yrange]
                       for x in xrange])
        plt.contour(xrange, yrange, grid,
                    (-1.0, 0.0, 1.0),
                    colors=('red', 'black', 'blue'),
                    linewidths=(1, 3, 1))

    def print_classification(self):
        xrange = np.arange(-5,8,0.5)
        yrange = np.arange(-5,8,0.5)
        points = [(x,y) for x in xrange for y in yrange]

        for point in points:
            if self.predict(point)==1:
                plt.plot(point[0],
                            point[1],
                            'go')
            else:
                plt.plot(point[0],
                            point[1],
                            'mo')
        plt.show()

if __name__ == '__main__':
    dataSet = []
    labels = []
    gt = Generator()
    
    data = gt.load_points("../data/svm_points.txt")  
    data = np.mat(data)
      
    dataSet = data[:,0:2]
    labels = data[:,2:3]
    train_x = dataSet[0:80,:]
    train_y = labels[0:80,:]
    test_x = dataSet[80:100,:]
    test_y = labels[80:100,:]
    classifier = SVM(C=1.0, epsilon=10**-5, kernel_option = ('linear',1.0,2), with_slack=False)
    classifier.train(train_x, train_y)
    #print classifier.support_vector
    classifier.print_data(train_x, train_y)
    classifier.print_boundaries()
    classifier.print_classification()
    acc = classifier.testsvm(test_x, test_y)
    print "The test accuracy: " + str(acc)