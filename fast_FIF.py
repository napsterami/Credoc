#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 13:40:37 2019

@author: amit
"""

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import math
import random
#%matplotlib inline
from matplotlib import pyplot as plt
import os
import time
import pandas as pd
import numpy as np # linear algebra
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
#% matplotlib inline




import seaborn as sns
import scipy.io
from sklearn import metrics
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import euclidean_distances
import re



random.seed()









class InNode:
    def __init__(self,left,right,splitAtt,splitVal,X_init):
        self.left=left
        self.right=right
        self.splitAtt=splitAtt
        self.splitVal=splitVal
        #self.mu_left=mu_left
        #self.mu_right=mu_right
        #self.mu_final=mu_final
        self.X_init=X_init
        
        
        
#  define for external Node

class ExNode:
    def __init__(self,size):
        self.size=size
        
        
        
# For Forest



def iForest(X,noOfTrees,sampleSize):
    forest=[]
    hlim=math.ceil(math.log(sampleSize,2))
    #hlim=len(X)
    #X_init = np.ones(len(X))
    for i in range(noOfTrees):
        #X_train=X.sample(sampleSize)
        forest.append(iTree(X,0,hlim,sampleSize))
    return forest


# Isolation Trees


def leftMF_array_NEW(q,X_l_LO_array,p_l,deno):
        #mvl = []
    s = re.search(r"\d+(\.\d+)?", q)
    num = int(s.group(0))
    mvl = (X_l_LO_array[num-1]-p_l)/deno
    return mvl


def rightMF_array_NEW(q,X_r_LO_array,p_r,deno):
    #mvr = []
    s = re.search(r"\d+(\.\d+)?", q)
    num = int(s.group(0))
    mvr = (p_r-X_r_LO_array[num-1])/deno
    return mvr




def compute_MF_ALL(q,MF,X_margin_array,deno,p_l,p_r):
    for i in range(len(X_margin_array)):
        if deno == 0:
            MF.append(1)
        else:
            MF.append(max(leftMF_array_NEW(q,X_margin_array[i],p_l,deno),rightMF_array_NEW(q,X_margin_array[i],p_r,deno)))
    return MF

def iTree(X,currHeight,hlim,sampleSize):
    if currHeight>=hlim or len(X)<=1:
        return ExNode(len(X))
    else:
        Q=X.columns    # list of attributes are in column
        
        q=random.choice(Q)  # choose a random attribute
        #random.seed(42)
        p=random.choice(X[q].unique())
        
        
        ratio = 0.3
        m = min(X[q])
        M = max(X[q])
        p_l = p - ((M - m) * ratio)
    
        p_r = p + ((M - m) * ratio)

        deno = p_r-p_l
         

        X_ll=X[X[q]<p]   # all the points on the left of the seperation line
        X_rr=X[X[q]>=p]  # all the points on the right of the seperation line
        
        
        X_margin = X[(X[q] >= p_l) & (X[q] <= p_r)] 
        X_margin_array = X_margin.to_numpy()
        
        MF = []
        MF = compute_MF_ALL(q,MF,X_margin_array,deno,p_l,p_r)
    
        
        X_init = np.ones(sampleSize)
        
        l = 0
        for i in X_margin.index:
            X_init[i] = MF[l]
            l+=1

         
        return InNode(iTree(X_ll,currHeight+1,hlim,sampleSize),iTree(X_rr,currHeight+1,hlim,sampleSize),q,p,X_init)

    

def findMU_ALL_NEW(i,x,Tree,res):
    if isinstance(Tree,ExNode):
        return 0
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:            #isinstance(Tree.left,InNode):
        return res.append(Tree.X_init[i]), findMU_ALL_NEW(i,x,Tree.left,res)
    else:                       #elif isinstance(Tree.right,InNode):
        return res.append(Tree.X_init[i]), findMU_ALL_NEW(i,x,Tree.right,res)




def tt(df_data,ifor,res):
    for i in range(len(df_data)):
    #pos=0
        for tree in ifor:
            res.append(findMU_ALL_NEW(i,df_data.iloc[i],tree,res))
        #print(i)
        #resss[i][j] = findMU(df_data.iloc[i],tree)
        #pos+=1
    return res









def checkType(a_list):
    flag=0
    count=[]
    for element in a_list:
        if isinstance(element, float):
            flag+=1
            
        if isinstance(element, tuple):
            count.append(flag)
            continue
        
    return count


def mfindex(res):
    #cc=checkType(res)
    
    
    coutput=[]
    
    for i in range(len(cc)-1):
        coutput.append(cc[i+1]-cc[i])
    
    coutput.insert(0, cc[0]) 
    
    
    res_ele=[x for x in res if not isinstance(x, tuple)]
    return res_ele,coutput

        
        

def mf_all_tree(coutput,res_ele,tree_size,df_data):
    offset = 0
    result_data = []
    for x in coutput:
        result_data.append(res_ele[offset:offset+x])
        offset += x
    
    
    result_data_tnorm=[]
    for i in range(len(result_data)):
        result_data_tnorm.append(round(np.prod(result_data[i]),4))
    
            
    final_res=[]
    
    j=0
    k=tree_size
    for i in range(len(df_data)):
        #temp=10
        final_res.append(result_data_tnorm[j:k])                           #  contains the list of forest for every element in input df_data 
        j=j+tree_size
        k=k+tree_size
    
    
    final_result=[]
    for i in range(len(final_res)):
        final_result.append(sum(final_res[i]))
    
    return final_result
    
 