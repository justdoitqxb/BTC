# -*- coding: utf-8 -*-
'''
Created on 2014-11-4

@author: qxb
'''
import numpy as np

def lr1(x,y):
    """y = ax + b"""
    avg_x = sum(x)/len(x)
    avg_y = sum(y)/len(y)
    x_sub = map((lambda x:x - avg_x), x)
    y_sub = map((lambda y:y - avg_y), y)
    x_sub_pow2 = map((lambda x: x**2), x_sub)
    x_y = map((lambda x,y: x * y), x_sub,y_sub)
    a  = sum(x_y) / sum(x_sub_pow2)
    b  = avg_y - a * avg_x
    return a,b

def nplr1(x,y):
    """y = ax + b"""
    avg_x = np.sum(x)/len(x)
    avg_y = np.sum(y)/len(y)
    x_sub = x - avg_x
    y_sub = y - avg_y
    x_sub_pow2 = x_sub**2
    x_y = x_sub * y_sub
    a  = np.sum(x_y) / np.sum(x_sub_pow2)
    b  = avg_y - a * avg_x
    return a,b

def nplrn(x,y):
    """y = b0 + b1x1 + b2x2 + ... + bkxk + e
    x--numpy.matrix
    y--numpy.matrix
    """
    b = (x.T * x).I*x.T*y
    print "parameter matrix is {0}".format(b)
    temp_e = y - x * b
    e = temp_e.sum() /temp_e.size
    print "bias is {0}".format(e)
    return b,e
    






