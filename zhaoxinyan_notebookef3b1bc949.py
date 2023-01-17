import pandas as pd

import numpy as np

import nltk, re, math, collections

from nltk.corpus import stopwords

from nltk.corpus import wordnet

import matplotlib.pylab as plt

import operator
train_v = pd.read_csv('../input/training_variants')

test_v = pd.read_csv('../input/test_variants')

train_t = pd.read_csv('../input/training_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])

test_t = pd.read_csv('../input/test_text',sep='\|\|',skiprows=1,engine='python',names=["ID","Text"])



train = pd.merge(train_v, train_t, how='left', on='ID').fillna('')

y_labels = train['Class'].values



test = pd.merge(test_v, test_t, how='left', on='ID').fillna('')

test_id = test['ID'].values
c1, c2, c3, c4, c5, c6, c7, c8, c9 = "", "", "", "", "", "", "", "", ""



for i in train[train["Class"]==1]["ID"]:

    c1+=train["Text"][i]+" "





for i in train[train["Class"]==2]["ID"]:

    c2+=train["Text"][i]+" "



for i in train[train["Class"]==3]["ID"]:

    c3+=train["Text"][i]+" "

    

for i in train[train["Class"]==4]["ID"]:

    c4+=train["Text"][i]+" "

    

for i in train[train["Class"]==5]["ID"]:

    c5+=train["Text"][i]+" "

    

    

for i in train[train["Class"]==6]["ID"]:

    c6+=train["Text"][i]+" "

    

for i in train[train["Class"]==7]["ID"]:

    c7+=train["Text"][i]+" "

    

for i in train[train["Class"]==8]["ID"]:

    c8+=train["Text"][i]+" "

    

    

for i in train[train["Class"]==9]["ID"]:

    c9+=train_t["Text"][i]+" "
def tokenize(_str):

    stops = set(stopwords.words("english"))

    tokens = collections.defaultdict(lambda: 0.)

    wnl = nltk.WordNetLemmatizer()

    for m in re.finditer(r"(\w+)", _str, re.UNICODE):

        m = m.group(1).lower()

        if len(m) < 2: continue

        if m in stops: continue

        if m.isnumeric():continue

        m = wnl.lemmatize(m)

        tokens[m] += 1 

    return tokens

texts_for_training=[]

texts_for_test=[]

num_texts_train=len(train)



print("Tokenizing training texts")

for i in range(0,num_texts_train):

    if((i+1)%1000==0):

        print("Text %d of %d\n"%((i+1), num_texts_train))

    texts_for_training.append(tokenize(train["Text"][i]))
print("Generating cluster 1")

cluster1=tokenize(c1)



print("Generating cluster 2")

cluster2=tokenize(c2)



print("Generating cluster 3")

cluster3=tokenize(c3)



print("Generating cluster 4")

cluster4=tokenize(c4)



print("Generating cluster 5")

cluster5=tokenize(c5)



print("Generating cluster 6")

cluster6=tokenize(c6)



print("Generating cluster 7")

cluster7=tokenize(c7)



print("Generating cluster 8")

cluster8=tokenize(c8)



print("Generating cluster 9")

cluster9=tokenize(c9)

def uniqsPerClass(clase, objective, exact):



    uniqs = collections.defaultdict(lambda: 0.)



    for t, v in clase.items():

        apears=0

        if t in cluster1:

            apears+=1

        if t in cluster2:

            apears+=1

        if t in cluster3:

            apears+=1

        if t in cluster4:

            apears+=1

        if t in cluster5:

            apears+=1

        if t in cluster6:

            apears+=1

        if t in cluster7:

            apears+=1  

        if t in cluster8:

            apears+=1

        if t in cluster9:

            apears+=1

    

        if exact:            

            if apears==objective:

                uniqs[t]=v

        else:

            if apears<(objective+1):

                uniqs[t]=v

    return uniqs

uniC1=uniqsPerClass(cluster1,1,False)

uniC2=uniqsPerClass(cluster2,1,False)

uniC3=uniqsPerClass(cluster3,1,False)

uniC4=uniqsPerClass(cluster4,1,False)

uniC5=uniqsPerClass(cluster5,1,False)

uniC6=uniqsPerClass(cluster6,1,False)

uniC7=uniqsPerClass(cluster7,1,False)

uniC8=uniqsPerClass(cluster8,1,False)

uniC9=uniqsPerClass(cluster9,1,False)

def termsComps(file):

    c1,c2,c3,c4,c5,c6,c7,c8,c9=0.,0.,0.,0.,0.,0.,0.,0.,0.

    for t, v in file.items():

        if t in uniC1:

            c1+=v

        if t in uniC2:

            c2+=v

        if t in uniC3:

            c3+=v

        if t in uniC4:

            c4+=v

        if t in uniC5:

            c5+=v

        if t in uniC6:

            c6+=v

        if t in uniC7:

            c7+=v

        if t in uniC8:

            c8+=v

        if t in uniC9:

            c9+=v

        suma=c1+c2+c3+c4+c5+c6+c7+c8+c9

        if suma==0:

            suma=1

            

    return [c1/suma,c2/suma,c3/suma,c4/suma,c5/suma,c6/suma,c7/suma,c8/suma,c9/suma]
uniqsTextMatr=[]

for file in texts_for_training:

    uniqsTextMatr.append(termsComps(file))



uniqText = pd.DataFrame(uniqsTextMatr, columns=['class'+str(c+1) for c in range(9)])

uniqText['RealClass'] = train["Class"]



uniqText.to_csv('uniqtrain.csv',index=False)