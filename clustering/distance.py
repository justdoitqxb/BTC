#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
from math import sqrt
def distance(vector1, vector2) :
    """
    Calculate distance between two vectors using pearson method
    """
    sum1 = sum(vector1)
    sum2 = sum(vector2)
    sum1Sq = sum([pow(v,2) for v in vector1])
    sum2Sq = sum([pow(v,2) for v in vector2])
    pSum = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
    num = pSum - (sum1*sum2/len(vector1))
    den = sqrt((sum1Sq - pow(sum1,2)/len(vector1)) * (sum2Sq - pow(sum2,2)/len(vector1)))
    if den == 0 : return 0.0
    return 1.0 - num/den