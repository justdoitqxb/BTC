#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
#AXA = A, XAX = X
#y = b1*x1 + b2*x2 + b3*(x1^2) + b4*(x2^2) + b5x1x2
#regression
import numpy as np

z = np.matrix([1.4,1.9,1.7,0.8,1.1]).T
print type(z)
x_t = np.matrix([[7,3],[3,17],[11,5]],dtype=np.float64)
x = np.matrix([[x_t[0,0],x_t[0,1],x_t[0,0]**2,x_t[0,1]**2,x_t[0,1]*x_t[0,0]],\
               [x_t[1,0],x_t[1,1],x_t[1,0]**2,x_t[1,1]**2,x_t[1,0]*x_t[1,1]],\
               [x_t[2,0],x_t[2,1],x_t[2,0]**2,x_t[2,1]**2,x_t[2,0]*x_t[2,1]]],\
              dtype=np.float64)
y = x * z
wn = np.linalg.pinv(x.T*x)
print wn
b = wn*x.T*y
print b
err = y - x*b
e = err.sum()/err.size
print e