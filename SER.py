# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:24:39 2021

@author: Maria
"""
import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def SERmodel_multneuro(NN,C,T,sp,refr,ia=1,d=0): 

## SER network simulation for multiple neurons in one region
# 
# NN           = Number of neurons in a region
# C            = Matrix of coupling (NxN) between pairs of regions (can be directed) 
# T            = Total time of simulated activity
# ia           = Initial condition setting. It can be a number representing the number of excited nodes 
#              (the remaining nodes are splitted in two equal size cohorts of susceptible and refractory nodes)
#              or can be a vector describing the initial state of each region
# d            = Initial time steps to remove (transient dynamics)
# 
# Convention is:
#       - susceptible node =  0
#       - excited node     =  1
#       - refractory node  = -1
##
        
    if T<=d:
        raise ValueError("Simulation time must be greater than the transient")
        
    if (len(ia)==1 and ia.item()>len(C)) or (len(ia)>len(C)):
        raise ValueError("Initial active nodes must be equal to or lower than the total number of nodes")

    numb=NN #number of neurons in a region
    N = len(C) #number of regions
    y = np.zeros((numb,N,T)).astype(np.int64) #initialize phase timeseries for one cycle
    
    f_inter = 1 #frequency of intermittent dbs, if =1 => no intermittency

    ##Initialization
    
    for j in range(numb):
        if len(ia)==N:
            y[j,:,0] = ia
        else:
            print('here')
            r4=np.random.choice(np.arange(N),ia.item(),replace=False)
            k4=y[j,:,0]
            k4[r4]=1
            y[j,:,0]=k4
            r5 = np.where(y[j,:,0]==0)[0]
            k5=y[j,:,0]
            k5[r5]=np.floor(1-2*np.random.uniform(0,1,len(r5)))
            y[j,:,0]=k5 
    

    for t in range(T-1):
        #updates for ser model
        for i in range(numb):

            r=(y[i,:,t]==1)
            k=y[i,:,t+1]
            k[r]=-1
            y[i,:,t+1]=k

            r1=np.logical_and(y[i,:,t]==0, (np.sum(C[:,y[i,:,t]==1],1)>0))
            k1=y[i,:,t+1]
            k1[r1]=1
            y[i,:,t+1]=k1

            r2=np.logical_and(np.logical_and(y[i,:,t]==0, (np.sum(C[:,y[i,:,t]==1],1)<=0)), (np.random.uniform(0,1,12)<sp))
            k2=y[i,:,t+1]
            k2[r2]=1
            y[i,:,t+1]=k2

            r3=np.logical_and(y[i,:,t]==-1, (np.random.uniform(0,1,12)<(1-refr)))
            k3=y[i,:,t+1]
            k3[r3]=-1
            y[i,:,t+1]=k3
    
    fin=y[:,:,d:]
    
    return fin  