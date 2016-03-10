'''Created on 2014-11-4
@author: qxb
'''
import numpy as np
import math

class LineClassifier:
    def __init__(self):
        self.b = 1
        self.a0 = 0.1
        self.a = 0.0
        self.r = 50.0
        self.expect_err = 0.05
        self.iter = 200
        self.testdata = []
    def testdata_init(self):
        self.testdata = []
    def e_init(self,err):
        self.expect_err = err
    def samples_init(self,samples):
        my_x = []
        my_y = []
        my_w = [self.b]
        examples = samples
        for example in examples:
            temp = [1] + example[0]
            my_x.append(temp)
            my_y.append(example[1])
        for i in range(len(my_x[0])-1):
            my_w.append(0.0)
        self.x = np.array(my_x)
        self.y = np.array(my_y)
        self.w = np.array(my_w)
    def a_init(self,mya):
        self.a0 = mya
    def r_init(self,myr):
        self.r = myr
    def maxtry_init(self,max_iter):
        self.iter = max_iter
    def sgn(self,v):
        if v>0:
            return 1
        else:
            return -1
    def get_v(self,myw,myx):
        return self.sgn(np.dot(myw.T,myx))
    def neww(self,oldw,myy,myx,a,iter_t):
        myerr = self.get_err(oldw,myx,myy)
        self.a = self.a0 / (1 + iter_t / float(self.r))
        return (oldw + a * myerr * myx,myerr)
    def get_err(self,myw,myx,myy):
        return myy - self.get_v(myw, myx)
    def train(self):
        iter_t = 0
        while True:
            myerr = 0
            i = 0
            for xn in self.x:
                self.w,err = self.neww(self.w, self.y[i], xn, self.a, iter_t)
                i+=1
                myerr+=pow(err,2)
            myerr = math.sqrt(myerr)
            iter_t+=1
            print "%d's time adjust error : %f"%(iter_t,myerr)
            if abs(myerr)<self.expect_err or iter_t > self.iter:
                if iter_t>self.iter:
                    print "reach the max iteration"
                break
    def simulate(self,testdata):
        if self.get_v(self.w,np.array([1]+testdata))>0:
            return 1
        else:
            return -1
       
            
            
            
            
            
            
             
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        