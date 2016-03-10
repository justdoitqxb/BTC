# -*- coding: utf-8 -*-
'''
Created on 2014-11-12

@author: qxb
'''
import time 

def timestamp_datetime(value):
    time_format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(time_format, value)
    return dt

def datetime_timestamp(dt):
    time.strptime(dt, '%Y-%m-%d %H:%M:%S')
    s = time.mktime(time.strptime(dt, '%Y-%m-%d %H:%M:%S'))
    return int(s)

def datetime2timestamp(dt):
    time.strptime(dt, u'%Y年%m月%d日%H:%M')
    s = time.mktime(time.strptime(dt, u'%Y年%m月%d日%H:%M'))
    return int(s)

if __name__ == '__main__':
    d1 = datetime_timestamp('2014-03-25 00:00:00')
    d2 = datetime_timestamp('2014-03-24 00:00:00')
    print d1 -d2
    ss = datetime2timestamp(u'2014年03月25日15:46')
    sss = datetime2timestamp(u'2014年03月08日12:31')
    print sss
    s = timestamp_datetime(1395733560)
    print s