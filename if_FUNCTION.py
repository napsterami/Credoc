#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:06:45 2020

@author: amit
"""

import isolation_forest_FUNCTION as iso

import time


start_time = time.time() 


TIME_FIF =[]
TIME =[]
PRECISION =[]
ACCURACY = []
RECALL = []



df_data=df
X = df
sampleSize=len(df)
tree_size=100                              # number of trees to be wanted in a forest
ifor=iso.iForest(df,tree_size,sampleSize) # ifor is the forest i.e. conatins list of "tree_size" trees 
print("--- %s seconds ---" % (time.time() - start_time))  
TIME_FIF.append(round((time.time() - start_time),4))
#print("--- %s seconds ---" % (time.time() - start_time)) 


start_time = time.time() 

FF,ll,pp=iso.pathlength_SIMPLE(df,ifor)   #  FF = average path length , ll = leaf, pp =path
FFF = iso.anomaly_score(FF,sampleSize)

FF2 = iso.anomaly_score_NEW(df,ifor)

df['anomalyscore'] = iso.anomaly_score(FF,sampleSize)


predicted_IF = [1 if x > 0.5 else 0 for x in df['anomalyscore']] 

#print("--- %s seconds ---" % (time.time() - start_time))  
TIME.append(round((time.time() - start_time),4))

confusion_mat(predicted_IF,labels, [0, 1])


#from sklearn.metrics import roc_auc_score
#roc_auc_score(predicted_IF,labels)



from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support)

fpr, tpr, thresholds = roc_curve(labels,FFF)
roc_auc = auc(fpr, tpr)
print("ROC curve value is:", round(roc_auc,2))



import bitarray as bt


a=map(int, labels)
int_a=[]
for intt in a:
   int_a.append(int(intt))
   
   
tp = (bt.bitarray(predicted_IF) & bt.bitarray(int_a)).count()
tn = (~bt.bitarray(predicted_IF) & ~bt.bitarray(int_a)).count()
fp = (bt.bitarray(predicted_IF) & ~bt.bitarray(int_a)).count()
fn = (~bt.bitarray(predicted_IF) & bt.bitarray(int_a)).count()



ACCURACY.append(round((tp + tn)/(tp + tn + fp + + fn),4) )
PRECISION.append(round(tp/(tp + fp),4))
RECALL.append(round(tp/(tp + fn),4))




df['IF_score'] = predicted_IF



plt.figure(figsize = (12,8))
plt.scatter(df['c1'], df['c2'], c=df['IF_score'], cmap='jet')
plt.xlabel('ISOLATION FOREST')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_YES--90--IF_C--0.22.png')
#plt.savefig('/home/amit/Documents/LibIsolationForest-master/python3/fuzzy isolation forest/Data4/HLIM_NO--70--IF_C--0.22.png')
plt.show()

df=df.drop(['anomalyscore'], axis=1)
df=df.drop(['IF_score'], axis=1)

# df=df.drop(['actual'], axis=1)
#subprocess.call(['speech-dispatcher'])        #start speech dispatcher
#subprocess.call(['spd-say', '"FINISHED"'])




print('\007')
