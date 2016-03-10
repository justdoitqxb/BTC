#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.

'''
import numpy as np
import jieba
import copy
class TextClassifier:
    def __init__(self):
        self.stop_words = []
        self.clicked_news = []
        self.predict_news = []
        
    def train_init(self,sample):
        self.clicked_news = sample
        
    def test_init(self,test):
        self.predict_news = test
            
    def set_stop_words(self):
        f_stop = open("data/stopwords.txt")
        try:
            f_stop_text = f_stop.read()
            f_stop_text = unicode(f_stop_text,'utf-8')
        finally:
            f_stop.close()
        self.stop_words = f_stop_text.split('\n')
        
    def get_cossimi(self,x,y):
        np_x = np.array(x)
        np_y = np.array(y)
        x_y = np.sum(np_x * np_y)
        x_x = np.sum(np_x * np_x)
        y_y = np.sum(np_y * np_y)
        if x_x == 0 or y_y ==0:
            return 0.0
        return x_y / float(np.sqrt(x_x)*np.sqrt(y_y))
    
    def split_text(self,content):
        ct = content.encode('utf-8')
        seg_list = jieba.cut(ct)#text divide into words
        return seg_list
    
    def generate_dict(self,seg_list):        
        # remove the words stopping to use
        test_words ={}
        for word in seg_list:
            #print ".", 
            if not(word.strip() in self.stop_words) and len(word.strip())>1:
                test_words.setdefault(word.strip(),0.0)
        return test_words
    
    def text_remove_a_count(self,seg_list,test_words):
        new_test_words = copy.deepcopy(test_words)
        test_vect = []
        for word in seg_list:
            #print ".",    
            if not(word.strip() in self.stop_words) and len(word.strip())>1:
                if word.strip() in new_test_words:
                    new_test_words[word.strip()]+=1
        for key in new_test_words:
            test_vect.append(new_test_words[key])
        return test_vect
    
    def linarweight(self,t_len):
        if t_len==0:
            return 0
        w = np.linspace(3,1,t_len,dtype=np.float64)
        w_sum = np.sum(w)
        w = w / w_sum
        return w
    
    def train(self,user_news,coarse_recomend,news_list):
        first = 0.0
        first_new = ''
        second = 0.0
        second_new = ''
        self.set_stop_words()
        self.train_init(user_news)
        self.test_init(coarse_recomend)
        test_word_list = []
        test_vect_list = []
        for new in self.clicked_news:            
            smp_seg_list = self.split_text(news_list[new][1]) 
            test_words = self.generate_dict(smp_seg_list)
            smp_seg_list = self.split_text(news_list[new][1])
            sample_vect = self.text_remove_a_count(smp_seg_list, test_words)
            test_word_list.append(test_words)
            test_vect_list.append(sample_vect)
        weight = self.linarweight(len(test_word_list))
        for n in self.predict_news:
            test_simi = 0.0
            for i in range(len(weight)):
                test_seg_list = self.split_text(news_list[n][1])
                test_vect = self.text_remove_a_count(test_seg_list, test_word_list[i])
                simi = self.get_cossimi(test_vect_list[i], test_vect)
                test_simi+=(weight[i]*simi)
            #print test_simi
            if test_simi > first:
                first = test_simi
                first_new = n
            else:
                if test_simi > second:
                    second = test_simi
                    second_new = n
        if second > first / 1.08:
            return first_new,second_new
        else:
            return first_new,""
 
      
if __name__ == '__main__':

    tc = TextClassifier()
    dict = {'a':[1,u'计算机计算机计算机游戏游戏基因'],'b':[2,u'人类去火星以前只是一个梦想现在已经可以登陆火星'],\
            'c':[3,u'计算机计算机计算机游戏游戏基因'],'d':[4,u'我想有一个计算机从而去超越梦想'],'e':[5,u'人与火星人是存在不可跨越的梦想计算机']}
    user_list = ['a','b']
    rc = ['c','d','e']
    print tc.train(user_list, rc, dict)    

