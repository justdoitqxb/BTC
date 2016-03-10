#!/usr/bin/env python
# encoding: utf-8
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import os,glob

def get_file_list(dir_path,extension):
    ''' func: et_file_list(dir_path,extension_list)
    params:
    dir_path-a string of path
    extension_list- a string of file extension eg 'txt'
    '''
    file_fullpath_list = []
    file_basename =[]
    extension = '*.'+extension
    full_name = dir_path + extension
    print full_name
    file_fullpath_list += [os.path.realpath(e) for e in glob.glob(full_name)]
    file_basename += [os.path.basename(p) for p in glob.glob(full_name)]
    return file_basename,file_fullpath_list
 
def get_fold_list(dir_path):
    ''' func: get_fold_list(dir_path)
    params:
    dir_path-a string of path
    return the fold in the dir_path
    '''   
    dir_list = []
    files = os.listdir(dir_path)
    for fl in files:
        if(os.path.isdir(dir_path + '\\' + fl)):
            #remove the hidden files
            if(fl[0] == '.'):
                pass
            else:
                dir_list.append(fl)
    return dir_list

if __name__ == '__main__':   
    dir_path = '..\\data\\'
    extension = 'txt'
    print get_file_list(dir_path, extension)
    