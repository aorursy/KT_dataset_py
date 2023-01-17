# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
df.head()
df1=df.drop(columns={'sepal_length','sepal_width','petal_width'},inplace=True)
df2=df.sort_values('petal_length')

df2
a=df2['petal_length'].unique()

a
def entropy(s,v,ver,all_total):

    if(s!=0):

        e1=-(s/all_total)*math.log2(s/all_total)

    else:

        e1=0

    if(v!=0):

        e2=-(v/all_total)*math.log2(v/all_total)

    else:

        e2=0

    if(ver!=0):

        e3=-(ver/all_total)*math.log2(ver/all_total)

    else:

        e3=0

    return(e1+e2+e3)
i=0

info_gain=[]

S=df2[df2['species']=='Iris-setosa'].shape[0]

V=df2[df2['species']=='Iris-virginica'].shape[0]

VER=df2[df2['species']=='Iris-versicolor'].shape[0]

total=df2.shape[0]

EP=-(S/total)*math.log2(S/total)-(V/total)*math.log2(V/total)-(VER/total)*math.log2(VER/total)

while i< len(a)-1:

    df_1=df2[df2['petal_length']>a[i]]

    S=df_1[df_1['species']=='Iris-setosa'].shape[0]

    V=df_1[df_1['species']=='Iris-virginica'].shape[0]

    VER=df_1[df_1['species']=='Iris-versicolor'].shape[0]

    total1=df_1.shape[0]

    E1=entropy(S,V,VER,total1)

    

    df_2=df2[df2['petal_length']<=a[i]]

    S1=df_2[df_2['species']=='Iris-setosa'].shape[0]

    V1=df_2[df_2['species']=='Iris-virginica'].shape[0]

    VER1=df_2[df_2['species']=='Iris-versicolor'].shape[0]

    total2=df_2.shape[0]

    E2=entropy(S1,V1,VER1,total2)

    WE=(total1/total)*E1 + (total2/total)*E2

    IGO=EP-WE

    info_gain.append(IGO)

    i+=1

indx=info_gain.index(max(info_gain))

a[indx]

    
## 1.9 is the best spliting criteria