#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 21:08:30 2020

@author: amit
"""


#  Run each segement to load the particular data (df) and its label (labels) 


#############################     Data D1     ##################################################### 

df = np.genfromtxt("Data8.csv", delimiter=",")
df=pd.DataFrame(df)
df.columns=['c1','c2']

labels = np.genfromtxt("Data8_labels.csv", delimiter=",")


plt.figure(figsize = (12,9))
plt.scatter(df['c1'], df['c2'])
plt.rcParams.update({'font.size': 14})
plt.grid(False)



############################     Data D2  ######################################################

df = np.genfromtxt("Donut.csv", delimiter=",")
df=pd.DataFrame(df)
df.columns=['c1','c2']

labels = np.genfromtxt("Donut_labels.csv", delimiter=",")

plt.figure(figsize = (12,9))
plt.scatter(df['c1'], df['c2'])
plt.rcParams.update({'font.size': 14})
plt.grid(False)


#########################       Data Ionosphere  #############################################


import scipy.io
mat = scipy.io.loadmat('ionosphere.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]





#########################       Data Arrhythmia  #############################################

import scipy.io
mat = scipy.io.loadmat('arrhythmia.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]



#########################       Data Pima  #############################################

df = np.genfromtxt("diabetes.csv", delimiter=",")
df=pd.DataFrame(df)
df = df.iloc[1:]

labels = df.iloc[:,8]
del df[8]

df.columns=['c1','c2','c3','c4','c5','c6','c7','c8']


df = df.reset_index()               # only for the data starting with index 1
df = df.drop(['index'],axis=1)




#########################       Data Glass  #############################################

import scipy.io
mat = scipy.io.loadmat('glass.mat')
df = mat['X']

labels = mat['y']
df=pd.DataFrame(df)
df.columns = ['c'+str(x) for x in range(1,len(df.columns)+1)]






#################   Data Compound    ##############################################

agg_data = np.loadtxt(fname = "Compound.txt")

agg_label=agg_data[:,2]



df_compound1 = agg_data[agg_data[:,2] ==1]
df_compound2 = agg_data[agg_data[:,2] ==2]


df_compound = np.vstack((df_compound1, df_compound2))

df_compound=pd.DataFrame(df_compound)
df_compound.columns=['c1','c2','labels']

df_compound['labels'][df_compound['labels'] == 2] = 0 


df_labels = df_compound['labels']
df_compound=df_compound.drop(['labels'], axis=1)

fig, ax = plt.subplots(figsize=(8,8))
plt.scatter(df_compound['c1'], df_compound['c2'], s=40, cmap='viridis')


df = df_compound.copy()
labels = df_labels.copy()


