'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import numpy as np

def linear_kernel():
    def kernel(matrix_x,sample_x):
        return matrix_x * sample_x.T + 1
    return kernel

def polynomial_kernel(k,degree):
    def kernel(matrix_x,sample_x):
        return np.power(matrix_x * sample_x.T + k,degree)
    return kernel

def sigmoid_kernel(k, delta):
    def kernel(matrix_x,sample_x):
        return np.tanh(k * matrix_x * sample_x.T - delta)
    return kernel

def rbf_kernel(sigma):
    def kernel(matrix_x,sample_x):            
        num_samples = matrix_x.shape[0]  
        xxh1 = np.dot(np.sum(np.power(matrix_x,2),1),np.ones((1,sample_x.shape[0]),np.float))
        xxh2 = np.dot(np.sum(np.power(sample_x,2),1),np.ones((1,matrix_x.shape[0]),np.float))
        omega = xxh1 + xxh2.T - 2.0 * (np.dot(matrix_x,sample_x.T))
        return np.exp(-omega/(2.0 * sigma**2))
    return kernel

def wavelet_kernel(k,degree,sigma):
    def kernel(matrix_x,sample_x):
        xxh1 = np.dot(np.sum(np.power(matrix_x,2),1),np.ones((1,sample_x.shape[0]),np.float))
        xxh2 = np.dot(np.sum(np.power(sample_x,2),1),np.ones((1,matrix_x.shape[0]),np.float))
        omega = xxh1 + xxh2.T - 2.0 * (np.dot(matrix_x,sample_x.T))
        
        xxh11 = np.dot(np.sum(matrix_x,1),np.ones((1,sample_x.shape[0]),np.float))
        xxh22 = np.dot(np.sum(sample_x,1),np.ones((1,matrix_x.shape[0]),np.float))
        omega1 = xxh11 -xxh22.T
        return np.multiply(np.cos(k * omega1/degree),np.exp(-omega/(2.0 * sigma**2)))
    return kernel

