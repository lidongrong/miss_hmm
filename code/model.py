# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 20:27:53 2022

@author: lidon
"""

from .learn import*
from .pred import*



class hmm_model:
    # a hidden markov model only receives hidden state and obs states as its initializer
    def __init__(self,hidden_state,obs_state,indicator='None'):
        #self.data=data
        self.hidden_state=hidden_state
        self.obs_state=obs_state
        # indicator is the indicator of missing
        self.indicator=indicator
        # initial distribution, transition matrix, emission matrix
        self.initial=None
        self.transition=None
        self.emission=None
        self.train_log=None
    
    # fit model on data
    # step: number of iterations that reports the condition
    # if choose not to report, step=0 (by default)
    # e: stopping threshold
    # core: number of cores to use, use mp.cpu_count() by default
    # learnt parameter will be saved in self.initial,self.emission and self.train_log
    # if choose to ouput training log, let step>0, and train log will be output to 
    # self.train_log
    def fit(self,data,step=0,e=0.001,core=None):
        data=data_parser(data,self.indicator)
        if step==0:
            a,b,initial,f=train(data,self.hidden_state,self.obs_state,step,e,core)
            self.initial=initial[len(initial)-1]
            self.transition=a[len(a)-1]
            self.emission=b[len(b)-1]
            return a,b,initial,f
        else:
            a,b,initial,f,log=train(data,self.hidden_state,self.obs_state,step,e,core)
            self.transition=a[len(a)-1]
            self.emission=b[len(b)-1]
            self.initial=initial[len(initial)-1]
            self.train_log=log
            return a,b,initial,f,log
    
    def predict(self,data,method='Viterbi'):
        data=data_parser(data,self.indicator)
        z=predict(data,self.transition,self.emission,self.initial,self.hidden_state,self.obs_state,method)
        return z
        
        
        