#!/usr/bin/env python
# encoding: utf-8
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import os,fnmatch,re

def get_file_list(tdir,pattern):
    '''
    func:get_file_list(dir,pattern)
    param:
        tdir-- a string of file path
        pattern -- a string of match file eg. '*.txt'
    '''
    file_list = []
    for file_name in os.listdir(tdir):
        if fnmatch.fnmatch(file_name, pattern):#select file matches the pattern
            #file_list += [file_name]
            file_list.append(file_name)
    return file_list

def get_file_list_multi(tdir,extension_list):
    '''
    func:get_file_list(dir,pattern)
    param:
        tdir-- a string of file path
        extension_list -- a string of match file eg. ['.txt','.data']
    '''   
    file_list = []
    for file_name in os.listdir(tdir):
        for ext in extension_list:
            ext = '*'+ext
            file_pattern = fnmatch.translate(ext)
            if re.match(file_pattern, file_name):
                file_list.append(file_name)
    return file_list
        


if __name__ == '__main__':
    print get_file_list('../data/news_class', '*.txt')[1].decode('GB18030')
    print get_file_list_multi('../data', ['.txt','.dat'])