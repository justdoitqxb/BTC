#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import loaddata, distance
import random

distances_cache = {}

def totalcost(blogwords, costf, medoids_idx) :
    size = len(blogwords)
    total_cost = 0.0
    medoids = {}
    for idx in medoids_idx :
        medoids[idx] = []
    for i in range(size) :
        choice = None
        min_cost = 2.1
        for m in medoids :
            tmp = distances_cache.get((m,i),None)
            if tmp == None :
                tmp = distance.distance(blogwords[m],blogwords[i])
                distances_cache[(m,i)] = tmp
            if tmp < min_cost :
                choice = m
                min_cost = tmp
        medoids[choice].append(i)
        total_cost += min_cost
    return total_cost, medoids
    

def kmedoids(blogwords, k) :
    size = len(blogwords)
    medoids_idx = random.sample([i for i in range(size)], k)
    pre_cost, medoids = totalcost(blogwords,distance.distance,medoids_idx)
    print pre_cost
    current_cost = 2.1 * size # maxmum of pearson_distances is 2.    
    best_choice = []
    best_res = {}
    iter_count = 0
    while 1 :
        for m in medoids :
            for item in medoids[m] :
                if item != m :
                    idx = medoids_idx.index(m)
                    swap_temp = medoids_idx[idx]
                    medoids_idx[idx] = item
                    tmp,medoids_ = totalcost(blogwords,distance.distance,medoids_idx)
                    #print tmp,'-------->',medoids_.keys()
                    if tmp < current_cost :
                        best_choice = list(medoids_idx)
                        best_res = dict(medoids_)
                        current_cost = tmp
                    medoids_idx[idx] = swap_temp
        iter_count += 1
        print current_cost,iter_count
        if best_choice == medoids_idx : break
        if current_cost <= pre_cost :
            pre_cost = current_cost
            medoids = best_res
            medoids_idx = best_choice
        
    
    return current_cost, best_choice, best_res

def print_match(best_medoids, blognames) :
    for medoid in best_medoids :
        print blognames[medoid],'----->',
        for m in best_medoids[medoid] :
            print '(',m,blognames[m],')',
        print
        print '---------' * 20 

if __name__ == '__main__' :
    blogwords, blognames = loaddata.load_data()
    best_cost,best_choice,best_medoids = kmedoids(blogwords,8)
    print_match(best_medoids,blognames)
