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
df=pd.read_csv('../input/play-tennis/play_tennis.csv')
df.head()
df.info()
col=df.columns

col
def entropy(y,n):

    if(y==0):

        return (-n*np.log2(n))

    elif (n==0):

        return (-y*np.log2(y))

    else:

        return (-y*np.log2(y)-n*np.log2(n))
all_entropy=[]





#entropy of parent

#E(P)=-py*log(py)-pn*log(pn)

yes=df[df['play']=='Yes']['play'].count()

no=df[df['play']=='No']['play'].count()

total=df.shape[0]

EP=-(yes/total)*math.log2(yes/total)-(no/total)*math.log2(no/total)

def each_col(x):

    col_index =df[x].value_counts().index

    all_entropy=[]



    i=0

    while i < len(col_index):

        df1=df[df[x]==col_index[i]]

        yes = df1[df1['play']=='Yes'].shape[0]

        no = df1[df1['play']=='No'].shape[0]

        p_y = yes/df1.shape[0]

        p_n = no/df1.shape[0]

        e = entropy(p_y,p_n)

        wt_entropy=(df1.shape[0]/total)*e

        all_entropy.append(wt_entropy)

        i+=1

    

    return (EP - sum(all_entropy))

col
i=1

info_gain={}

while i < len(col)-1:

    val = each_col(col[i])

    colm = col[i]

    info_gain[colm]=val

    i=i+1
info_gain


print(' The first split will take place on {} column '.format(max(info_gain,key=info_gain.get)))