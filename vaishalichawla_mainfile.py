# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from os import makedirs
from os.path import join, exists, expanduser
from glob import glob
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# Any results you write to the current directory are saved as output.
data_path = '../input/data/'
images = []
df = pd.read_csv(os.path.join(data_path,'Data_Entry_2017.csv'))

df2 = df[['Image Index','Finding Labels']]
#print(df2.at[104999, 'Image Index'])
df2 = df2[104999:112120][:]

image_paths = glob(os.path.join(data_path,'images_012','images','*.png'))

#print(image_paths[image_paths.index(os.path.join(data_path, 'images_004','images',df2['Image Index'][29999]))])
# #all_image_paths = { os.path.basename(x): x for x in image_paths }


for i in range(104999,112120):
    img = df2['Image Index'][i]
    path = image_paths[image_paths.index(os.path.join(data_path, 'images_012','images', img))]
    #print(path)
    temp = cv2.imread(path)
    im = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    images.append(im)
print(len(images))
import math
def idm(g):
    r, c = g.shape
    idm = 0.0
    for i in range(r):
        for j in range(c):
            idm += (1/(1+(i-j)**2)) * g[i][j]
    return idm

def variance(g):
    r, c = g.shape
    var = 0.0
    for i in range(r):
        for j in range(c):
            var += ((i-j)**2)*g[i][j]
    return var

def sumaver(g, rowsum, colsum):
    P = rowsum + colsum
    sa = 0.0
    for i in range(len(P)):
        sa += i*P[i]
    return sa

def sumentr(g, rowsum, colsum):
    P = rowsum + colsum
    se = 0.0
    for i in range(len(P)):
        if(P[i]>0):
            se -= P[i]*(math.log(P[i]))
    return se

def diffentr(g, rowsum, colsum):
    P = rowsum-colsum
    de=0.0
    for i in range(len(P)):
        if(P[i]>0):
            de -= P[i]*(math.log(P[i]))
    return de

def inertia(g):
    r, c = g.shape
    iner = 0.0
    for i in range(r):
        for j in range(c):
            iner += ((i-j)**2)*g[i][j]
    return iner

def clus_shade_prom(g, rowmean, colmean):
    shade = 0.0
    prom = 0.0
    r, c = g.shape
    for i in range(r):
        for j in range(c):
            shade += ((i+j-rowmean-colmean)**3)*g[i][j]
            prom += ((i+j-rowmean-colmean)**4)*g[i][j]
    return shade, prom

def entropy(g):
    r,c = g.shape
    entr = 0.0
    for i in range(r):
        for j in range(c):
            if(g[i][j]!=0):
                entr += g[i][j] * (math.log(g[i][j]))
    return -1*entr
import skimage.feature
import numpy as np
contrast=[0]*len(images)
homogen=[0]*len(images)
energy=[0]*len(images)
correl=[0]*len(images)
dissim=[0]*len(images)
asm=[0]*len(images)
indm=[0]*len(images)
var=[0]*len(images)
sumav=[0]*len(images)
sument=[0]*len(images)
dent=[0]*len(images)
iner=[0]*len(images)
shade=[0]*len(images)
prom=[0]*len(images)
entr=[0]*len(images)
maxprob=[0]*len(images)
rowsums=[0]*len(images)
colsums=[0]*len(images)

for i in range(len(images)):
    g = skimage.feature.greycomatrix(images[i], [1], [0], levels=256, symmetric=False, normed=True)
  
    contrast[i] = skimage.feature.greycoprops(g, 'contrast')[0][0]
    homogen[i] = skimage.feature.greycoprops(g, 'homogeneity')[0][0]
    energy[i] = skimage.feature.greycoprops(g, 'energy')[0][0]
    correl[i] = skimage.feature.greycoprops(g, 'correlation')[0][0]
    dissim[i] = skimage.feature.greycoprops(g, 'dissimilarity')[0][0]
    asm[i] = skimage.feature.greycoprops(g, 'ASM')[0][0]
    r, c, d, t = g.shape
    g = np.reshape(g, [r, c])

    rowsums[i] = np.sum(g, axis=1)
    colsums[i] = np.sum(g, axis=0)

    rowmean = 0
    colmean = 0
    assert r==c

    rowmean = np.sum(j*rowsums[i][j] for j in range(r))
    colmean = np.sum(j*colsums[i][j] for j in range(c))

    indm[i] = idm(g)
    var[i] = variance(g)
    sumav[i] = sumaver(g, rowsums[i], colsums[i])
    sument[i] = sumentr(g, rowsums[i], colsums[i])
    dent[i] = diffentr(g, rowsums[i], colsums[i])
    iner[i] = inertia(g)
    s, p = clus_shade_prom(g, rowmean, colmean)
    shade[i] = s
    prom[i] = p
    entr[i] = entropy(g)
    maxprob[i] = np.max(g)
df2['Contrast']=contrast
df2['Homogeneity']=homogen
df2['Energy']=energy
df2['Correlation']=correl
df2['Dissimilarity']=dissim
df2['ASM']=asm
df2['Inverse Difference Moment']=indm
df2['Variance']=var
df2['Sum Averange']=sumav
df2['Sum Entropy']=sument
df2['Difference Entropy']=dent
df2['Inertia']=iner
df2['Cluster Shade']=shade
df2['Cluster Prominence']=prom
df2['Entropy']=entr
df2['Maximum Probability']=maxprob
df2.head()
labelset = df2['Finding Labels']
labels=[]
for l in labelset:
    s = l.split('|')  #spliting to get the list of labels associated with each instance
    for item in s:
        labels.append(item)  #add the labels so obtained to the list of labels
labels=set(labels)       #making the list a set, to get unique values
#print(labels)
#creating a separate column in the data frame for each label
for x in labels:
    df2[x] = [0]*len(images)
#to assign 0/1 to each label column, if occurring in that image or not
for a in range(104999, 112120):
    label_set = df2['Finding Labels'][a].split('|')
    for l in label_set:
        df2[l][a] = 1
df2.head()
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if(feature_name == 'Image Index' or feature_name == 'Finding Labels'):
            continue
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result
df2.to_csv('Features_5k_21.csv')
