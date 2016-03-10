'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import numpy as np

def sigmoid():
    def action_func(x):
        return 1 / (1 + np.exp(-x))
    return action_func

def sine():
    def action_func(x):
        return np.sin(x)
    return action_func

def hardlim():
    def action_func(x):
        x[x>=0] = 1
        x[x<0] = 0        
        return x
    return action_func

def rbfnet():
    def action_func(x):      
        return np.exp(x)
    return action_func