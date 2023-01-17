import numpy as np

import pandas as pd



df = pd.read_csv('../input/xigua30/xigua_data3.0.csv',index_col=0)
df.loc[df['密度']>df['密度'].median(),'密度']='高'

df.loc[df['密度']!='高','密度']='低'

df.loc[df['含糖率']>df['含糖率'].median(),'含糖率']='高'

df.loc[df['含糖率']!='高','含糖率']='低'



label=df['好瓜']=='是'

df=df.drop(['好瓜'],axis=1)

df
label
from math import log



def entropy(data,label):

    ent = 0

    for l in label.unique():

        p = sum(label==l)/len(label)

        ent-= p*log(p,2)

    return ent
class node:

    def __init__(self,df,label):

        self.data=df

        self.label=label

        

        self.criterion=None

        self.children=[]

        self.ent = entropy(df,label)

        self.leaf=-1

        self.type=''

        

    def divide(self):

        if len(self.label.unique())==1:

            self.leaf=list(self.label)[0]

            return

        if len(self.data.columns)==0:

            self.leaf=self.label.value_counts().index[0]

            return

        

        #寻找最优属性

        IGs={}

        for c in self.data.columns:

            IG=self.ent

            for typ in self.data[c].unique():

                ent = entropy(self.data.loc[self.data[c]==typ],self.label[self.data[c]==typ])

                IG -= ent*sum(self.data[c]==typ)/len(self.data[c])

            IGs[IG]=c

            

        if max(IGs.keys())>0:

            self.criterion=IGs[max(IGs.keys())]

            c=self.criterion

            for typ in self.data[c].unique():

                #建立子节点

                new_data = self.data.loc[self.data[c]==typ].drop(c,axis=1)

                new_label = self.label[self.data[c]==typ]

                new = node(new_data,new_label)

                new.type=str(typ)

                new.divide()

                self.children.append(new)

        else:

            self.leaf=self.label.value_counts().index[0]

            return
TREE = node(df,label)

TREE.divide()
def bfs(node):

    if node.criterion:

        print("按%s划分"%node.criterion)

        for c in node.children:

            print(c.type,end='\t')

        print()

        for c in node.children:    

            print(c.leaf,end='\t')

        print()    

        for c in node.children:

            if c.leaf==-1:print('如果%s%s-->'%((node.criterion,c.type)),end='')

            bfs(c)

        
bfs(TREE)
with open('readme','w') as f:

    f.write('ID3算法划分西瓜')