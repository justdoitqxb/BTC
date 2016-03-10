#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
def load_data(FIFE = '../data/blogdata.txt') :
    blogwords = []
    blognames = []
    f = open(FIFE, 'r') 
    words = f.readline().split()
    #//remove '\r\n'
    for line in f:    
        blog = line[:-2].split('\t')
        blognames.append(blog[0])        
        blogwords.append([int(word_c) for word_c in blog[1:]]       ) 
    return blogwords,blognames
