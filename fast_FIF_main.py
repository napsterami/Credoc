#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:24:05 2020

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
#import isolation_forest_FUZZY_ONE_FUNCTION as fiso
import fast_FIF as fiso

import time
import subprocess

start_time = time.time() 


TIME_FIF =[]
TIME =[]
PRECISION =[]
ACCURACY = []
RECALL = []
ROC = []
THRESH = []


PRECISION2 =[]
ACCURACY2 = []
RECALL2 = []
ROC2 = []
THRESH2 = []



#df=df.drop(['anomalyscore'], axis=1)




"""

"""


df_data=df       #  Assign the input Dataframe to the variable df_data for which you wants the simIF
X=df_data


sampleSize=len(df_data)
tree_size=100                           # number of trees to be wanted in a forest
ifor=fiso.iForest(df_data,tree_size,sampleSize) # ifor is the forest i.e. conatins list of "tree_size" trees 
print("--- %s seconds ---" % (time.time() - start_time))  
TIME_FIF.append(round((time.time() - start_time),4))
#print("--- %s seconds ---" % (time.time() - start_time)) 


start_time = time.time() 
res=[]

rest=fiso.tt(df_data,ifor,res)


cc=fiso.checkType(rest)


coutput=[]
    
for i in range(len(cc)-1):
    coutput.append(cc[i+1]-cc[i])
    
coutput.insert(0, cc[0]) 
res_ele=[x for x in rest if not isinstance(x, tuple)]


temp_res=fiso.mf_all_tree(coutput,res_ele,tree_size,df)

  
FINAL_RESULT = [x / tree_size for x in temp_res]


df['mf_value'] = FINAL_RESULT



TH =0.2                  #######  Threshold Value


pred = [1 if x >= TH else 0 for x in FINAL_RESULT]

#print("--- %s seconds ---" % (time.time() - start_time))  
#TIME.append(round((time.time() - start_time),4))


df['mf_score'] = pred



#print("--- %s seconds ---" % (time.time() - start_time))  






#################    Fuzzy IF             --------------------------------------------------------------------


plt.figure(figsize = (12,8))
plt.scatter(df['c1'], df['c2'], c=df['mf_score'], cmap='jet')
plt.xlabel('FUZYY IF')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_YES--100--FuzzyIF.png')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_NO--70--FuzzyIF.png')
plt.show()

confusion_mat(pred,labels, [0, 1])



import bitarray as bt


a=map(int, labels)
int_a=[]
for intt in a:
   int_a.append(int(intt))
   
   
tp = (bt.bitarray(pred) & bt.bitarray(int_a)).count()
tn = (~bt.bitarray(pred) & ~bt.bitarray(int_a)).count()
fp = (bt.bitarray(pred) & ~bt.bitarray(int_a)).count()
fn = (~bt.bitarray(pred) & bt.bitarray(int_a)).count()

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)


fpr, tpr, thresholds = roc_curve(labels,FINAL_RESULT)
roc_auc = auc(fpr, tpr)
print("ROC curve value is:", round(roc_auc,2))
ROC.append(roc_auc)




ACCURACY.append(round((tp + tn)/(tp + tn + fp + + fn),4) )
PRECISION.append(round(tp/(tp + fp),4))
RECALL.append(round(tp/(tp + fn),4))

"""
thresh = []
for i in df.tail(5).index:
    thresh.append(FINAL_RESULT[i])
    
THRESH.append(min(thresh))


T = min(thresh)

#sort_index = np.argsort(FINAL_RESULT)

#sort_list = sorted(FINAL_RESULT)


df_label = pd.DataFrame(labels)
df_label.columns=['c']
df_index = df_label.index[df_label['c']==1]

FINALL = []
for i in df_index:
    FINALL.append(FINAL_RESULT[i])
    
THRESH.append(min(FINALL))
T =min(FINALL)

MIN_of_IRREGULAR = T


FI = [i for i in FINAL_RESULT if i >= T]







SCORES =pd.DataFrame(FINAL_RESULT)

SCORES1 = SCORES[ SCORES.iloc[:,0]>= TH ]    # all the point which are higher than the threhold; have many FP 


#df['labels'] = labels

RR = df_label[df_label['c'] == 0]
IRR = df_label[df_label['c'] == 1]

#REGULAR = df[df['labels'] == 0] # all the regular points in the dataset
#IRREGULAR = df[df['labels'] == 1] # all the regular points in the dataset
#REGULAR_INDEX = REGULAR.index
RR_INDEX = RR.index
SCORES1_INDEX = SCORES1.index




COMMON  = SCORES1_INDEX[SCORES1_INDEX.isin(RR_INDEX)].dropna()

REGULAR_SCORES = []
for i in COMMON:
    REGULAR_SCORES.append((SCORES1.iloc[SCORES1.index == i]).to_numpy()) 



REGULAR_SCORES1 = np.concatenate(REGULAR_SCORES, axis=0)     # scores of all the normal points in the list of all the points after threhold
SCORES1 = SCORES1.to_numpy()


print("Minimum score of the Iregular points::", min(SCORES1))
print("Maximum score of the Iregular points::", max(SCORES1))
print("---------------------------------------------------")
print("Minimum score of the Regular points above the Threshold", TH, "::", min(REGULAR_SCORES1))
print("Maximum score of the Regular pointsabove the Threshold", TH, "::", max(REGULAR_SCORES1))


per_reg = round(((len(REGULAR_SCORES1) * 100)/len(RR)),2)


print("Percentage of the Regular points above the threshold: ", per_reg , "%")





#df=df.drop(['labels'], axis=1)




pred = [1 if x >= TH else 0 for x in FINAL_RESULT]


plt.figure(figsize = (12,8))
plt.scatter(df['c1'], df['c2'], c=df['mf_score'], cmap='jet')
plt.xlabel('FUZYY IF')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_YES--100--FuzzyIF.png')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_NO--70--FuzzyIF.png')
plt.show()

confusion_mat(pred,labels, [0, 1])
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_YES--90--FuzzyIF-Confusion Matrix.png')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_NO--70--FuzzyIF-Confusion Matrix.png')



import bitarray as bt


a=map(int, labels)
int_a=[]
for intt in a:
   int_a.append(int(intt))
   
   
tp = (bt.bitarray(pred) & bt.bitarray(int_a)).count()
tn = (~bt.bitarray(pred) & ~bt.bitarray(int_a)).count()
fp = (bt.bitarray(pred) & ~bt.bitarray(int_a)).count()
fn = (~bt.bitarray(pred) & bt.bitarray(int_a)).count()


fpr, tpr, thresholds = roc_curve(labels,FINAL_RESULT)
roc_auc = auc(fpr, tpr)
print("ROC curve value is:", round(roc_auc,2))
ROC2.append(roc_auc)




ACCURACY2.append(round((tp + tn)/(tp + tn + fp + fn),4) )
PRECISION2.append(round(tp/(tp + fp),4))
RECALL2.append(round(tp/(tp + fn),4))

"""


df=df.drop(['mf_value'], axis=1)
df=df.drop(['mf_score'], axis=1)

#df=df.drop(['actual'], axis=1)

#subprocess.call(['speech-dispatcher'])        #start speech dispatcher
#subprocess.call(['spd-say', '"Sawan main lag gayi aag ke dil mera haaaaye"'])



print('\007')





"""

scores = [i-TH for i in FINAL_RESULT]



from sklearn.ensemble import IsolationForest

clf = IsolationForest( max_samples=256,
                      random_state=0, contamination='auto')
clf.fit(df)
scores_dec = clf.decision_function(df)
scores = (-1.0) * clf.decision_function(df) 

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(labels,scores)
roc_auc = auc(fpr, tpr)
print("ROC curve value is:", round(roc_auc,2))
import matplotlib.pyplot as plt
plt.plot(fpr, tpr, 'k-', lw=2)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()









outlier_detect = IsolationForest(n_estimators=100, max_samples=256, contamination= 'auto', random_state = 42,behaviour='new')
outlier_detect.fit(df)
outliers_predicted = outlier_detect.predict(df)

#scores_pred = outlier_detect.decision_function(X)

pred_IF= outliers_predicted
pred_IF = [1 if x == -1 else 0 for x in pred_IF] 

df['IF_score'] = pred_IF


plt.figure(figsize = (12,8))
plt.scatter(df['c1'], df['c2'], c=df['IF_score'], cmap='jet')
plt.xlabel('ISOLATION FOREST')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_YES--90--IF_C--0.1.png')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_NO--70--IF_C--0.1.png')
plt.show()


confusion_mat(pred_IF,labels, [0, 1])

df=df.drop(['IF_score'], axis=1)





"""