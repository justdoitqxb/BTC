#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import random
import pickle
import math
import matplotlib.pyplot as plt

class Generator:
    
    def generate_2d_func_points(self, number, mu, sigma):
        # y = sin(x) / x , x~=0
        # y = 1, x = 0
        points = []
        for i in range(number):
            x = random.uniform(-10,10)
            if x == 0.0:
                y_x = 1.0 + random.gauss(mu, sigma)
            else:
                y_x = math.sin(x)/x +random.gauss(mu, sigma)
            points.append([x,y_x])
        return points
    
    def generate_2d_points(self,number,label, x_mean, x_var, y_mean, y_var):
        return [[random.normalvariate(x_mean,x_var),random.normalvariate(y_mean,y_var),label] for i in range(number)]
    
    def store_points(self, points, filename):
        with open(filename,'w') as outfile:
            pickle.dump(points, outfile)
    
    def load_points(self,filename):
        with open(filename,'r') as infile:
            l = pickle.load(infile)
        return l
    
    def plot_data(self,points):
        for point in points:
            if point[2] == 1:
                plt.plot(point[0],point[1],'bo')
            else:
                plt.plot(point[0],point[1],'ro')
                
    def plot_points(self,points):
        for point in points:
            plt.plot(point[0],point[1],'bo')
                
if __name__ == "__main__":
    gt = Generator()
    '''
    generate_new = False
    if generate_new:
        positive_points = gt.generate_2d_points(50, 1, 1, 0.5, 1, 0.5)
        negative_points = gt.generate_2d_points(50, 0, 3, 0.5, 3, 0.5)
        data = negative_points+positive_points
        random.shuffle(data)
        gt.store_points(data, "../data/elm_points.txt")
    else:
        data = gt.load_points("../data/elm_points.txt")
    '''
    points = gt.generate_2d_func_points(100, 0, 0.01)
    gt.store_points(points, "../data/reg_points.txt")
    
    gt.plot_points(points) 
    plt.show()   