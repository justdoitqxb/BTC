#!/usr/bin/env python
#-*- coding: UTF-8 -*-
'''
@author:     qxb
@copyright:  2014 organization_name. All rights reserved.
@license:    license
@contact:    user_email
@deffield    updated: Updated
'''
import numpy as np
import jieba,chardet
import copy
import sys,os
sys.path.append("..")
import ioio.txtoperater as to
class TextClassifier:
    def __init__(self):
        self.stop_words = []
        self.train_txt_path = ""
        self.test_txt_path = ""
        self.classDict = {}
        self.classNum = 1
    def train_path_init(self,tpath):
        self.train_txt_path = tpath
    def test_path_init(self,ttpath):
        self.test_txt_path = ttpath
    def init_class_dict(self):
        file_list = to.get_file_list(self.train_txt_path, '*.txt')
        self.classNum = len(file_list)
        for fl in file_list:
            fl = fl.decode('GB18030')
            print 'deal with ' + fl + '......'
            fl = self.train_txt_path + '/' + fl
            seg_list = self.split_text(fl) #list
            dict_words_count ={}
            for word in seg_list:
                if not(word.strip() in self.stop_words) and len(word.strip())>1:
                    dict_words_count.setdefault(word.strip(),0.0)
                    dict_words_count[word.strip()]+=1
            self.get_prior(dict_words_count)
            filename = os.path.basename(fl).split('.')[0]
            print filename
            self.classDict.setdefault(filename,dict_words_count)            
    def set_stop_words(self):
        f_stop = open("../res/stopwords.txt")
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
        return x_y / float(np.sqrt(x_x)*np.sqrt(y_y))
    def split_text(self,filename):
        f = open(filename) #open the labeled file
        try:
            fl_text = f.read()
            #fl_text = unicode(fl_text,'utf-8') #change the encoding
            txtencode = chardet.detect(fl_text).get('encoding','utf-8')
            fl_text = fl_text.decode(txtencode,'ignore').encode('utf-8')
        finally:
            f.close()
        seg_list = jieba.cut(fl_text)#text divide into words
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
                if new_test_words.has_key(word.strip()):
                    new_test_words[word.strip()]+=1
        for key in new_test_words:
            test_vect.append(new_test_words[key])
        return test_vect
    
    def get_prior(self,txt_word):
        allword = 0.0;
        basegl = 1e-8
        for key in txt_word:
            allword += txt_word[key]
        for key in txt_word:
            if allword > 0:
                txt_word[key] = basegl + txt_word[key] / allword
            else:
                txt_word[key] = basegl
    def get_post(self,txt_word,test_word,numb):
        itag = 0
        postprob = 1.0 / numb
        basegl = 1e-8
        for key in test_word:
            if txt_word.has_key(key):
                postprob *= test_word[key]
                postprob *= txt_word[key]
            else:
                postprob *= basegl
            if postprob < 1e-100:
                itag+=1
                postprob*=1e50                
        return postprob,itag   
    def train_cos_simi(self,labeld_file,test_file):
        tc.set_stop_words()
        smp_seg_list = tc.split_text(labeld_file) 
        test_words = tc.generate_dict(smp_seg_list)
        smp_seg_list = tc.split_text(labeld_file) 
        test_seg_list = tc.split_text(test_file)

        sample_vect = tc.text_remove_a_count(smp_seg_list, test_words)
        test_vect = tc.text_remove_a_count(test_seg_list, test_words)
        testsimi = tc.get_cossimi(sample_vect, test_vect) 
        print testsimi.encode('utf-8')
        return testsimi
    def train_pp(self,label_dir,test_file):
        test_class = ''
        max_pro = 0.0
        tag = float('Inf')       
        tc.set_stop_words()
        tc.train_path_init(label_dir)
        tc.init_class_dict()
        test_seg_list = tc.split_text(test_file)
        test_dict = {}
        for word in test_seg_list:
            if not(word.strip() in self.stop_words) and len(word.strip())>1:
                test_dict.setdefault(word.strip(),0.0)
                test_dict[word.strip()]+=1
        for key in self.classDict:
            post,itag = self.get_post(self.classDict[key], test_dict, self.classNum)
            print key,post,itag
            if itag <= tag and post > max_pro:
                max_pro = post
                tag = itag
                test_class = key
        print 'predict is : ' + test_class
        return test_class
        
        
        
if __name__ == '__main__':
    '''   
    tc = TextClassifier()
    cos_simi1 = tc.train_cos_simi("../data/APEC1.txt","../data/APEC2.txt")
    cos_simi2 = tc.train_cos_simi("../data/APEC1.txt","../data/other.txt")    
    '''
    tc = TextClassifier()
    tc.train_pp("../data/news_class", "../data/other.txt")