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
from copy import copy
  
class Hmm(object):
    
    def __init__(self,A,B,pi,I = np.array([]),O = np.array([]),criterion = 10**-4,precision = np.float):
        self.A = A    # transition probability matrix 
        self.B = B    # the observation probability
        self.pi = pi  # initial probability vector like [0.5,0.5]
        self.I = I    # state sequence
        self.O = O    # observation sequence
        self._eta = self._eta1
        self.criterion = criterion
        self.precision = precision
        
    def _reset(self,n, m, init_type = "uniform"):
        # baum balch not suit for uniform
        if init_type == "uniform":
            self.A = np.ones((n,n),self.precision) * (1.0 / n)
            self.B = np.ones((n,m),self.precision) * (1.0 / m)
            self.pi = np.ones((n),self.precision) * (1.0 / n)
        elif init_type == 'random':
            self.A = self._normalize_row(np.random.rand(n,n))
            self.B = self._normalize_row(np.random.rand(n,m))
            self.pi = np.random.random(n)
            self.pi = self.pi / self.pi.sum()
            
    def _normalize_row(self,matrix):
        shape = matrix.shape
        result = np.zeros(shape,self.precision)
        for i in xrange(shape[0]):
            s = matrix[i,0:shape[1]].sum()
            result[i,0:shape[1]] = matrix[i,0:shape[1]] / np.float(s)
        return result
        
    def _forward(self):
        # initialize 
        num_state = len(self.B)  #???????????????????????
        T = len(self.O)
        aerfa = np.zeros((T,num_state),self.precision)
        for i in xrange(num_state):
            aerfa[0,i] = self.pi[i] * self.B[i,self.O[0]]
        # recursion
        for t in xrange(T - 1):
            for j in xrange(num_state):
                for k in xrange(num_state):
                    aerfa[t+1,j] += aerfa[t,k] * self.A[k,j]
                aerfa[t+1,j] *= self.B[j,self.O[t+1]]
        # computer P(O|lamda)
        P = np.sum(aerfa[T-1,:])   
        return P,aerfa
        
    def _backward(self):
        #initialize: 
        num_state = len(self.B)  #???????????????????????
        T = len(self.O)
        beta=np.zeros((T,num_state),self.precision) 
        for i in xrange(num_state): 
            beta[T-1,i]=1  
  
        #recursion:  
        for t in xrange(T-2,-1,-1):  
            for j in xrange(num_state):  
                for k in xrange(num_state):  
                    beta[t,j] += self.A[j,k] * self.B[k,self.O[t+1]] * beta[t+1,k]  
         
        P_back = 0
        for l in xrange(num_state): 
            P_back += self.pi[l] * self.B[l,self.O[0]] * beta[0,l]
            
        return P_back,beta
    
    def _viterbi(self):
        # initialize
        
        num_state = len(self.B)
        T = len(self.O)
        delta = np.zeros((T,num_state),self.precision)
        w = np.zeros((T,num_state),self.precision)
        I = np.zeros(T)
        for i in xrange(num_state):
            delta[0,i] = self.pi[i] * self.B[i,self.O[0]]
            w[0,i] = 0
        # recursion
        for t in xrange(1,T):
            for j in xrange(num_state):
                delta[t,j]=self.B[j,self.O[t]]*np.array([delta[t-1,k]*self.A[k,j]  for k in xrange(num_state)]).max() 
                w[t,j]=np.array([delta[t-1,l]*self.A[l,j]  for l in xrange(num_state)]).argmax() 
      
        #termination  
        P=delta[T-1,:].max()  
        I[T-1]=delta[T-1,:].argmax()  
        for t in xrange(T-2,-1,-1):  
            I[t]=w[t+1,I[t+1]]  
        return I,P
    
    def _decode(self):
        '''
        Find the best state sequence (path), given the model and an observation. i.e: max(P(Q|O,model)).
        
        This method is usually used to predict the next state after training. 
        '''        
        # use Viterbi's algorithm. It is possible to add additional algorithms in the future.
        return self._viterbi()
    
    def viterbi_again(self, observations):
        '''
        Find the best state sequence (path) using viterbi algorithm - a method of dynamic programming,
        very similar to the forward-backward algorithm, with the added step of maximization and eventual
        backtracing.
        
        delta[t][i] = max(P[q1..qt=i,O1...Ot|model] - the path ending in Si and until time t,
        that generates the highest probability.
        
        psi[t][i] = argmax(delta[t-1][i]*aij) - the index of the maximizing state in time (t-1), 
        i.e: the previous state.
        '''
        # similar to the forward-backward algorithm, we need to make sure that we're using fresh data for the given observations.
        self._mapB(observations)
        num_state = len(self.B)
        delta = np.zeros((len(observations),num_state),self.precision)
        psi = np.zeros((len(observations),num_state),self.precision)
        
        # init
        for x in xrange(num_state):
            delta[0][x] = self.pi[x]*self.B_map[x][0]
            psi[0][x] = 0
        
        # induction
        for t in xrange(1,len(observations)):
            for j in xrange(num_state):
                for i in xrange(num_state):
                    if (delta[t][j] < delta[t-1][i]*self.A[i][j]):
                        delta[t][j] = delta[t-1][i]*self.A[i][j]
                        psi[t][j] = i
                delta[t][j] *= self.B_map[j][t]
        
        # termination: find the maximum probability for the entire sequence (=highest prob path)
        p_max = 0 # max value in time T (max)
        path = np.zeros((len(observations)),self.precision)
        for i in xrange(num_state):
            if (p_max < delta[len(observations)-1][i]):
                p_max = delta[len(observations)-1][i]
                path[len(observations)-1] = i
        
        # path backtracing
#        path = numpy.zeros((len(observations)),dtype=self.precision) ### 2012-11-17 - BUG FIX: wrong reinitialization destroyed the last state in the path
        for i in xrange(1, len(observations)):
            path[len(observations)-i-1] = psi[len(observations)-i][ path[len(observations)-i] ]
        return path
    
    def _mapB(self,observations):
        '''
        Required implementation for _mapB. Refer to _BaseHMM for more details.
        '''
        num_state = len(self.B)
        self.B_map = np.zeros( (num_state,len(observations)), self.precision)
        
        for j in xrange(num_state):
            for t in xrange(len(observations)):
                self.B_map[j][t] = self.B[j][observations[t]]
    
    def _eta1(self,t,T):
        '''
        Governs how each sample in the time series should be weighed.
        This is the default case where each sample has the same weigh, 
        i.e: this is a 'normal' HMM.
        '''
        return 1.
    
    def _gamma(self, aerfa=None, beta=None):
        #generate gamma as part of your update step
        T = len(self.O)
        num_state = len(self.B)
        if aerfa is None:
            P,aerfa = self._forward()
        if beta is None:
            P_back,beta = self._backward()
        gamma = np.zeros((T,num_state),self.precision)
        #use forward and backward data to get gamma
        for t in range(0,T):
            for i in range(0,num_state):
                abit = aerfa[t,i] * beta[t,i]
                s = 0
                for j in range(0,num_state):
                    s = s + (aerfa[t,j] * beta[t,j])
                    pass
                gamma[t,i] = abit/s
                pass
            pass
        return gamma
    
    def _calcxi(self,alpha=None,beta=None):
        '''
        Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.
        
        The xi variable is a numpy array indexed by time, state, and state (TxNxN).
        xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
        time 't+1' given the entire observation sequence.
        ''' 
        self._mapB(self.O)
        num_state = len(self.B)       
        if alpha is None:
            P,alpha = self._forward()
        if beta is None:
            P_back,beta = self._backward()
        xi = np.zeros((len(self.O),num_state,num_state),self.precision)
        
        for t in xrange(len(self.O)-1):
            denom = 0.0
            for i in xrange(num_state):
                for j in xrange(num_state):
                    thing = 1.0
                    thing *= alpha[t][i]
                    thing *= self.A[i][j]
                    thing *= self.B_map[j][t+1]
                    thing *= beta[t+1][j]
                    denom += thing
            for i in xrange(num_state):
                for j in xrange(num_state):
                    numer = 1.0
                    numer *= alpha[t][i]
                    numer *= self.A[i][j]
                    numer *= self.B_map[j][t+1]
                    numer *= beta[t+1][j]
                    xi[t][i][j] = numer/denom              
        return xi
    
    def _calcgamma(self,xi):
        '''
        Calculates 'gamma' from xi.
        
        Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
        in state 'i' at time 't' given the full observation sequence.
        '''   
        T = len(self.O)  
        num_state = len(self.B)   
        gamma = np.zeros((T,num_state),self.precision)
        
        for t in xrange(T):
            for i in xrange(num_state):
                gamma[t][i] = sum(xi[t][i])        
        return gamma
    
    def _updateA(self,xi,gamma):
        '''        
        Returns A_new, the modified transition matrix. 
        '''
        T = len(self.O)  
        num_state = len(self.B) 
        newA = np.zeros((num_state,num_state),self.precision)
        for i in xrange(num_state):
            for j in xrange(num_state):
                numer = 0.0
                denom = 0.0
                for t in xrange(T-1):
                    numer += (self._eta(t,T-1)*xi[t][i][j])
                    denom += (self._eta(t,T-1)*gamma[t][i])
                newA[i][j] = numer/denom
        return newA
    
    def _updateB(self,gamma):
        '''
        Helper method that performs the Baum-Welch 'M' step
        for the matrix 'B'.
        '''        
        T = len(self.O) 
        m = self.B.shape[1] 
        num_state = len(self.B) 
        # TBD: determine how to include eta() weighing
        newB = np.zeros( (num_state,m) ,self.precision)
        
        for j in xrange(num_state):
            for k in xrange(m):
                numer = 0.0
                denom = 0.0
                for t in xrange(T):
                    if self.O[t] == k:
                        numer += gamma[t][j]
                    denom += gamma[t][j]
                newB[j][k] = numer/denom
        
        return newB
    
    def _baum_welch(self,iter = 200):
        num_state = len(self.B)  #???????????????????????
        #T = len(self.O)  
        for it in xrange(iter):            
            # forward
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            P,aerfa = self._forward() 
            # backward
            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            P_back,beta = self._backward()
            xi = self._calcxi(aerfa, beta)  
            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = self._gamma(aerfa, beta)
            # Need final gamma element for new B
            newpi = gamma[1,0:num_state]
            newA = self._updateA(xi, gamma)
            newB = self._updateB(gamma)

            if np.max(np.abs(self.pi - newpi)) < self.criterion and np.max(np.abs(self.A - newA)) < self.criterion and np.max(np.abs(self.B - newB)) < self.criterion:
                break
            
            self.A[:] = newA
            self.B[:] = newB
            self.pi[:] = newpi  
            
    def train(self):

        nStates = self.A.shape[0]
        T = len(self.O)

        A = self.A
        B = self.B
        pi = copy(self.pi)
        
        done = False
        while not done:
            # alpha_t(i) = P(O_1 O_2 ... O_t, q_t = S_i | hmm)
            # Initialize alpha
            alpha = np.zeros((nStates,T))
            c = np.zeros(T) #scale factors
            alpha[:,0] = pi.T * self.B[:,self.O[0]]
            c[0] = 1.0/np.sum(alpha[:,0])
            alpha[:,0] = c[0] * alpha[:,0]
            # Update alpha for each observation step
            for t in range(1,T):
                alpha[:,t] = np.dot(alpha[:,t-1].T, self.A).T * self.B[:,self.O[t]]
                c[t] = 1.0/np.sum(alpha[:,t])
                alpha[:,t] = c[t] * alpha[:,t]

            # beta_t(i) = P(O_t+1 O_t+2 ... O_T | q_t = S_i , hmm)
            # Initialize beta
            beta = np.zeros((nStates,T))
            beta[:,T-1] = 1
            beta[:,T-1] = c[T-1] * beta[:,T-1]
            # Update beta backwards from end of sequence
            for t in range(T-1,0,-1):
                beta[:,t-1] = np.dot(self.A, (self.B[:,self.O[t]] * beta[:,t]))
                beta[:,t-1] = c[t-1] * beta[:,t-1]

            xi = np.zeros((nStates,nStates,T-1));
            for t in range(T-1):
                denom = np.dot(np.dot(alpha[:,t].T, self.A) * self.B[:,self.O[t+1]].T,
                               beta[:,t+1])
                for i in range(nStates):
                    numer = alpha[i,t] * self.A[i,:] * self.B[:,self.O[t+1]].T * \
                            beta[:,t+1].T
                    xi[i,:,t] = numer / denom
  
            # gamma_t(i) = P(q_t = S_i | O, hmm)
            gamma = np.squeeze(np.sum(xi,axis=1))
            # Need final gamma element for new B
            prod =  (alpha[:,T-1] * beta[:,T-1]).reshape((-1,1))
            gamma = np.hstack((gamma,  prod / np.sum(prod))) #append one more to gamma!!!
                         
            newpi = gamma[:,0]
            newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
            newB = copy(B)
            
            numLevels = self.B.shape[1]
            sumgamma = np.sum(gamma,axis=1)
            for lev in range(numLevels):
                mask = self.O == lev
                newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma

            if np.max(abs(pi - newpi)) < self.criterion and \
                   np.max(abs(A - newA)) < self.criterion and \
                   np.max(abs(B - newB)) < self.criterion:
                done = 1;
  
            A[:],B[:],pi[:] = newA,newB,newpi

        self.A[:] = newA
        self.B[:] = newB
        self.pi[:] = newpi
        
    def _simulate(self, n_step, num):
        def init_locate(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]
        observations = np.zeros((num, n_step),self.precision)
        states = np.zeros((num, n_step),self.precision)
        for j in xrange(num):
            states[j,0] = init_locate(self.pi)
            observations[j,0] = init_locate(self.B[states[j,0],:])
            for t in xrange(1,n_step):
                states[j,t] = init_locate(self.A[states[j,t-1],:])
                observations[j,t] = init_locate(self.B[states[j,t],:])
        return observations,states
    
    def _simulate_one(self, n_step):
        def init_locate(probs):
            return np.where(np.random.multinomial(1,probs) == 1)[0][0]
        observations = np.zeros(n_step)
        states = np.zeros(n_step)
        states[0] = init_locate(self.pi)
        observations[0] = init_locate(self.B[states[0],:])
        for t in xrange(1,n_step):
            states[t] = init_locate(self.A[states[t-1],:])
            observations[t] = init_locate(self.B[states[t],:])
        return observations,states
        
        
if __name__ == "__main__":

    #input  Matrix A,B vector pi  
    A = np.array([[0.8, 0.1, 0.1],[0.2, 0.4, 0.4],[0.3, 0.2, 0.5]])
    B = np.array([[0.8, 0.2],[0.5, 0.5],[0.4, 0.6]])
    #O=np.array([0 ,1,0,0,1,0])#T=3  
    #O=np.array([1 ,0, 1])#T=3  
    pi = np.array([0.2, 0.3, 0.5])  
    hmm = Hmm(A,B,pi) 
    hmm.O,hmm.I = hmm._simulate_one(50)
    
    #print hmm._backward()
    #print hmm._forward()
    print hmm.I
    print hmm._viterbi()
    #print hmm._gamma()
    #print hmm._calcgamma(hmm._calcxi())
    
    hmmt = Hmm(A,B,pi,O=hmm.O,I=hmm.I)
    hmmt._reset(3, 2, init_type='random')
    
    hmmt.O,hmmt.I = hmm._simulate_one(1000)
    hmmt.train()
    #print hmmt._calcxi()
    print hmmt.A
    print hmmt.B
    print hmmt.pi
    
