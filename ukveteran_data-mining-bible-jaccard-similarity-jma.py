import numpy as np
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import re

%matplotlib inline
index=pd.read_csv('../input/bible/bible_version_key.csv')
index
asv = pd.read_csv('../input/bible/t_asv.csv')
bbe = pd.read_csv('../input/bible/t_bbe.csv')
dby = pd.read_csv('../input/bible/t_dby.csv',encoding='latin-1')
kjv = pd.read_csv('../input/bible/t_kjv.csv')
wbt = pd.read_csv('../input/bible/t_wbt.csv')
web = pd.read_csv('../input/bible/t_web.csv')
ylt = pd.read_csv('../input/bible/t_ylt.csv')
ct=asv.groupby(['b'])['t'].count()
plt.bar(range(1,67),ct)
counts = dict()
for text in asv['t']:
    tokens=text.lower().split()
    tokens=[re.sub(r'[^\w\s]','',i) for i in tokens]
    for i in tokens: 
        if i in counts:
            counts[i]+=1
        else:
            counts[i]=1
sorted_counts = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
print("10 most common words:\nWord\tCount")
for word, count in sorted_counts[:10]:
    print("{}\t{}".format(word, count))

print("\n10 least common words:\nWord\tCount")
for word, count in sorted_counts[-10:]:
    print("{}\t{}".format(word, count))
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
#Book index for Corinthians, Chapter and verse number
b,c,vn=46,13,4

diff=pd.DataFrame(index['version'])
ver=[asv,bbe,dby,kjv,wbt,web,ylt]
for i,v in enumerate(ver):
    diff.loc[[i],'verse'] =v[(v['b']==b) & (v['c']==c) &(v['v']==vn)]['t'].values
    
diff
#Compare two verses
a=diff.loc[[0],'verse'].to_string(index=False)
b=diff.loc[[4],'verse'].to_string(index=False)
get_jaccard_sim(a,b)
#Metric Matrix
jac=pd.DataFrame(index=range(7))
for it in range(7):
    jac[it]=[get_jaccard_sim(diff.loc[[it],'verse'].to_string(index=False),
                          diff.loc[[i],'verse'].to_string(index=False)) for i in range(7)]

sns.heatmap(jac, annot=True)