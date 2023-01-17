# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import itertools



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#load data

train=pd.read_csv('../input/train.csv')

del train['id']
#create a Dataframe with duplicate question ids

ids_duplicate=train[train.is_duplicate==1][["qid1", "qid2"]]

#replace qid1 and qid2 by min(qid1,qid2) and max(qid1,qid2)

ids_duplicate['qidmin']=ids_duplicate.apply(lambda x:min(x['qid1'],x['qid2']),axis=1)

ids_duplicate['qidmax']=ids_duplicate.apply(lambda x:max(x['qid1'],x['qid2']),axis=1)

del ids_duplicate["qid1"]

del ids_duplicate["qid2"]

#create a string key

ids_duplicate['key']=ids_duplicate.apply(lambda x:str(x.qidmin)+' '+str(x.qidmax), axis=1)



#create a Dataframe with different questions ids

ids_different=train[train.is_duplicate==0][["qid1", "qid2"]]

#replace qid1 and qid2 by min(qid1,qid2) and max(qid1,qid2)

ids_different['qidmin']=ids_different.apply(lambda x:min(x['qid1'],x['qid2']),axis=1)

ids_different['qidmax']=ids_different.apply(lambda x:max(x['qid1'],x['qid2']),axis=1)

del ids_different["qid1"]

del ids_different["qid2"]

#create a string key

ids_different['key']=ids_different.apply(lambda x:str(x.qidmin)+' '+str(x.qidmax), axis=1)



#build a question DB

basequestion=np.concatenate((np.array(train[['qid1','question1']]), np.array(train[['qid2','question2']])),axis=0)

basequestion=pd.DataFrame(basequestion, columns=["q", "question"])

basequestion=basequestion.drop_duplicates()

basequestion.index=basequestion['q']

basequestion.head()
#Time to identify new duplicates and new differents

new_duplicates=[]

new_differents=[]

for qidmin in ids_duplicate.qidmin.unique():

    #for each qidmin value, we'll get a list of their duplicates and differents

    list_dup=list(ids_duplicate[ids_duplicate.qidmin==qidmin]["qidmax"])

    list_dif=list(ids_different[ids_different.qidmin==qidmin]["qidmax"])

    #each dup1 and dup2 from list_dup is a duplicate, so:

    for dup1,dup2 in itertools.combinations(list_dup,2):

        if dup1<dup2:

            dup1,dup2= dup2,dup1

        #we can add [dup1, dup2] as a potentially new duplicate

        new_duplicates.append(str(dup1)+' '+str(dup2))

        for dif in list_dif:

            #then for each element of list_dif, we can say that dup1 and dup2 are different of this element

            #because they're duplicate of qidmin

            new_differents.append(str(min(dif,dup1))+' '+str(max(dif,dup1)))

            new_differents.append(str(min(dif,dup2))+' '+str(max(dif,dup2)))



#now let's drop "duplicates":

new_duplicates=list(set(new_duplicates)-set(ids_duplicate['key']))

print(len(new_duplicates), "new duplicates")

new_differents=list(set(new_differents)-set(ids_different['key']))

print(len(new_differents), "new different")
#build new train data

train_add_list=[]

for elt in new_duplicates:

    q1,q2= elt.split(" ")

    q1=int(q1)

    q2=int(q2)

    train_add_list.append([q1, q2, basequestion.ix[q1].question, basequestion.ix[q2].question, 1])

for elt in new_differents:

    q1,q2= elt.split(" ")

    q1=int(q1)

    q2=int(q2)

    train_add_list.append([q1, q2, basequestion.ix[q1].question, basequestion.ix[q2].question, 0])

train_add=pd.DataFrame(train_add_list, columns=train.columns)

train_add.sample(20)
#Export data as a new train set

pd.concat([train, train_add], axis=0).to_csv('newtrain.csv')