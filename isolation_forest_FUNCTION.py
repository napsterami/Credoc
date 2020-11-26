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


start_time = time.time()  


random.seed()







class InNode:
    def __init__(self,left,right,splitAtt,splitVal):
        self.left=left
        self.right=right
        self.splitAtt=splitAtt
        self.splitVal=splitVal
        
        
        
#  define for external Node

class ExNode:
    def __init__(self,size):
        self.size=size

        
        
        
# For Forest


def iForest(X,noOfTrees,sampleSize):
    forest=[]
    hlim=math.ceil(math.log(sampleSize,2))
    #hlim=len(X)
    for i in range(noOfTrees):
        X=X.sample(sampleSize)
        forest.append(iTree(X,0,hlim))
    return forest


# Isolation Trees

#random.seed(42)
def iTree(X,currHeight,hlim):
    if currHeight>=hlim or len(X)<=1:
        return ExNode(len(X))
    else:
        
        Q=X.columns    # list of attributes are in column
        
        q=random.choice(Q)  # choose a random attribute
        #random.seed(42)
        p=random.choice(X[q].unique())
        
     
        
        X_l = X[X[q]<p]
        X_r = X[X[q]>=p]
        return InNode(iTree(X_l,currHeight+1,hlim),iTree(X_r,currHeight+1,hlim),q,p)


# Path Length   
    


    
def FindParent(x,Tree):
    #if isinstance(Tree,InNode):
    if isinstance(Tree,ExNode):
    #if type(Tree) == ExNode:
        return 0
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        return Tree.splitAtt, FindParent(x,Tree.left)
    else:
        return Tree.splitAtt, FindParent(x,Tree.right)
    


#sampleSize=len(df)
#tree_size=100                             # number of trees to be wanted in a forest
#ifor=iForest(df,tree_size,sampleSize) # ifor is the forest i.e. conatins list of "tree_size" trees 




        
        """
        att_range = np.arange(X.shape[1])
        att_list = random.sample(list(att_range), 2)
        
        
        dist_qp = dict()
        for q in att_list:
            p = np.random.uniform(X.iloc[:, q].min(), X.iloc[:, q].max())
            size = min(len(X[X.iloc[:, q] < p]), len(X[X.iloc[:, q] > p]))
            dist_qp[size] = (q, p)
        tuple = dist_qp[min(dist_qp.keys())]
        split_att = tuple[0]
        split_val = tuple[1]

        if split_att == 0:
            q = X.columns[0]
        else:
            q = X.columns[1]
            
        p =split_val
        """   




def pathLength(x,Tree,CurrentPathLength,ele):
    if isinstance(Tree,ExNode):
        return CurrentPathLength + c(ele)
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        return pathLength(x,Tree.left,CurrentPathLength+1,ele)
    else:
        return pathLength(x,Tree.right,CurrentPathLength+1,ele)
      



def pathLength_NEW(x,Tree,CurrentPathLength):
    while isinstance(Tree, InNode):
        if x[Tree.splitAtt] < Tree.splitVal:
            Tree = Tree.left
            CurrentPathLength += 1
        else:
            Tree = Tree.right
            CurrentPathLength += 1
    path_length_x = CurrentPathLength + c(Tree.size)
    return path_length_x
    

def anomaly_score_NEW(X,ifor):

        #if isinstance(X, pd.DataFrame):
        #    X = X.values
    cn = c(len(X))

    path_lengths = []
    for x in X.iloc:
        for tree in ifor:
            path_lengths.append(pathLength_NEW(x, tree,0))

    path_matrix = np.asarray(path_lengths).reshape(len(X), -1)
    Eh = path_matrix.mean(axis=1)

    s = 2 ** (-1.0 * Eh / cn)
    return s





def pathlengthh(df,ifor,elements):
    #actual_path=[]
    actual_path_length=[]
    
    
    for i in range(len(df)):
        itr=0
        for tree in ifor:
            
            #actual_path.append(FindParent(df.iloc[i],tree))          # 
            actual_path_length.append(pathLength(df.iloc[i],tree,0,elements[itr]))
            itr+=1
            #print(itr)
            
    return actual_path_length


##pp,pp1=pathlength(df_data,ifor)
    


def pathlength_SIMPLE(df,ifor):
    paths = []
    for row in df.iloc:
            #print(row)
        path = []
        leaf = []
        for tree in ifor:
                #node = tree.root
            length = 0
            while isinstance(tree, InNode):
                a=tree.splitAtt
                if row[a] < tree.splitVal:
                    tree = tree.left
                else:
                    tree = tree.right
                length += 1
            leaf_size = tree.size
            pathLength = length + c(leaf_size)
            leaf.append(leaf_size)   # to check  the expernal nodes presnet after item is identified in a tree
            path.append(pathLength)
        paths.append(path)
    paths = np.array(paths)
        
    return np.mean(paths, axis=1),leaf,paths


def pathlengthALL(df,ifor,pp1,tree_size):
    #all_ele_forest=[]
    all_ele_forest_actual_path_length=[]
    
    j=0
    k=tree_size
    for i in range(len(df)):
        #temp=10
        #all_ele_forest.append(actual_path[j:k])                           #  contains the list of forest for every element in input df_data 
        all_ele_forest_actual_path_length.append(pp1[j:k]) #  contains the list of path length of the trees for every element in input df_data 
        j=j+len(ifor)
        k=k+len(ifor)
        
    return all_ele_forest_actual_path_length
    #temp=temp+10


#pp2=pathlengthALL(df_data,ifor,pp1)


def Findleafelement(x,Tree,no_of_elements):
    if isinstance(Tree,ExNode):
        no_of_elements.append(Tree.size)
        return no_of_elements
    #elif isinstance(Tree.left,InNode):
    a=Tree.splitAtt
    if x[a]<Tree.splitVal:
        Findleafelement(x,Tree.left,no_of_elements) #,Findleafelement(Tree.right,no_of_elements)
        return no_of_elements
    else:
        Findleafelement(x,Tree.right,no_of_elements)#, Findleafelement(Tree.left,no_of_elements)
        return no_of_elements
    

#sum(i > 1 for i in no_of_elements)

#no_of_elements=[]
#Findleafelement(ifor[0])
#greater = [i for i in no_of_elements if i > 1]
#elements=sum(greater)-len(greater)
#print(elements)

def external_node_count(ifor):

    #no=1
    #all_val=[]
    for k in range(len(ifor)):
        no_of_elements=[]
        ress=Findleafelement(ifor[k],no_of_elements)
        greater = [i for i in ress if i > 1]
        elements=sum(greater)-len(greater)
        
        #no+=1
    
    return elements






















def c(n):
    if n > 2:
        return 2.0*(np.log(n-1)+0.5772156649) - (2 * (n - 1) / n)
    elif n == 2:
        return 1
    else:
        return 0






def pathlength_SUMALL(pp2):
    path=[]
    for i in range(len(pp2)):
        path.append(sum(pp2[i]))
    return path
    
#PP=pathlength_SUMALL(pp2)

  
#FINAL_PATH = [x / tree_size for x in PP]   # E(h(x))


def anomaly_score(inp,sampleSize):

    #score=[]
    for i in range(len(inp)):
        #score.append(pow(2,-(FINAL_PATH[i]/c_factor(tree_size))))
        #score.append(2.0**(-inp[i]/c(len(inp))))
        scores = np.array([np.power(2, -l/c(sampleSize)) for l in inp])
    return scores
        



"""
def Findleafelement(Tree):
    if isinstance(Tree,ExNode):
        return no_of_elements.append(Tree.size)
    elif isinstance(Tree.left,InNode):
        return Findleafelement(Tree.left), Findleafelement(Tree.right)
    else:
        return Findleafelement(Tree.right), Findleafelement(Tree.left)
"""