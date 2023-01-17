# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read all files 

import os 

fd = open('/kaggle/input/Names_data_train_large.txt',mode='r',encoding='UTF-8-sig')



#split tag and name

tag = []

name = []

for line in fd.readlines():

    tag.append(line[:1])

    #get rid of the '\n' after name

    name.append(line[2:-1])

    

if(len(tag)==len(name)):

    print('column match!')

    

print(name)

#feature engineering, load data into pd dataframe and visualize them

d = {'tag':tag,'name':name}

train_data = pd.DataFrame(data=d)





#first initiative:name length-tag

train_data['name-length'] = [len(n) for n in train_data['name']]

#print(train_data)

#plot it 

q=[]

for t in train_data['tag']:

    if(t == '+'):

        q.append(1)

    else:

        q.append(0)

train_data['tag-idx'] = q

#train_data.plot(y='name-length',x='tag-idx',kind='scatter')

#but the pos and negtive distribution looks evenly, but pos will beat neg a little, indicating the length is not a good classifier

#ty to generate more information from

#find one outlier with name length 50 ty to delete it

for i,l in enumerate(train_data['name-length']):

    if(l > 40):

        print(i)

#print(train_data.loc[187])

#get Ludwig Freiherr von und zu der Tann-Rathsamhausen after google it i find out its a german general, so its not random name

#this gives us a little hint about the background of the dataset, by using the a-priori knowledge, we can infer that the people

#with certain name section like "von" will have more pos? check how many people has name with "von"

count=0

for n in train_data['name']:

    try:

        if n.index(' von ') != 0:

            print(n)

    except:

        pass

#we can only got 2 names, not enough to describe the trend

#Christian Leopold von Buch

#Ludwig Freiherr von und zu der Tann-Rathsamhausen
#inspired by the first letters of names





def letter2idx(x):

    return ord(x.lower()) - 96

r = [n[:1] for n in train_data['name']]

i = map(letter2idx,r)

train_data['first-letter-idx']=list(i)

print(train_data)



#turn name into one-hot encoding

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1,1))

name = vectorizer.fit_transform(train_data["name"])

#prepare for test vocabulary

train_features = vectorizer.get_feature_names()

nameOneHot = pd.DataFrame(name.toarray(),columns=vectorizer.get_feature_names())

df = pd.concat([train_data,nameOneHot],axis=1)



#ty plot it

train_data.plot(x='first-letter-idx',y='tag-idx',kind='scatter')
#get dummies

#d = {'tag':tag,'name':name}

#test = pd.DataFrame(data=d)

#newt = pd.get_dummies(test)

#newt
#define x and y

#x_label = newt.keys().drop(['tag_+','tag_-'])

#target_label = 'tag_+'

#x_label



#new label

x_label_new = df.keys().drop(['name','tag','tag-idx'])

target_label_new = 'tag-idx'



#scale the data for better convergence

x_train = df[x_label_new]

y_train = df[target_label_new]





from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(x_train)

x_train = scaler.transform(x_train)



df[x_label_new]

#Common Model Algorithms

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process

from xgboost import XGBClassifier



#Common Model Helpers

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from sklearn import feature_selection

from sklearn import model_selection

from sklearn import metrics

#save model for test set

import pickle

MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model.RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

    tree.ExtraTreeClassifier(),

    

    #Discriminant Analysis

    discriminant_analysis.LinearDiscriminantAnalysis(),

    discriminant_analysis.QuadraticDiscriminantAnalysis(),



    

    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html

    XGBClassifier()    

    

    ]







#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit

#note: this is an alternative to train_test_split

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%



#create table to compare MLA metrics

MLA_columns = ['MLA Name','MLA Test Accuracy Mean']

MLA_compare = pd.DataFrame(columns = MLA_columns)



#df[target_label_new] = df[target_label_new][:-1]





#saved model

saved = {}

#index through MLA and save performance to table

row_index = 0

for alg in MLA:



    #set name and parameters

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name  

    

    

    #save MLA predictions - see section 6 for usage

    alg.fit(x_train,df[target_label_new])

    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = alg.score(x_train,df[target_label_new]) 

    

    #save model for test purpose

    saved[MLA_name] = pickle.dumps(alg)

    

    row_index+=1

 

    #print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html



MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)

MLA_compare

#MLA_predict
#choose the best 

for i in range(len(MLA_compare)):

    bestn = MLA_compare.loc[i]['MLA Name']



    best = pickle.loads(saved[bestn])



    #df_test = pd.read_fwf('/kaggle/input/Names_data_test.txt',names=["Class", "Name"])

    fd = open('/kaggle/input/Names_data_test.txt',mode='r',encoding='UTF-8-sig')



    #split tag and name

    tagt = []

    namet = []

    for line in fd.readlines():

        tagt.append(line[:1])

        #get rid of the '\n' after name

        namet.append(line[2:-1])

    

    #feature engineering, load data into pd dataframe and visualize them

    d = {'tag':tagt,'name':namet}

    test_data = pd.DataFrame(data=d)





#

    test_data['name-length'] = [len(n) for n in test_data['name']]

#print(train_data)

#plot it 

    q=[]

    for t in test_data['tag']:

        if(t == '+'):

            q.append(1)

        else:

            q.append(0)

    test_data['tag-idx'] = q



    def letter2idx(x):

        return ord(x.lower()) - 96

    r = [n[:1] for n in test_data['name']]

    i = map(letter2idx,r)

    test_data['first-letter-idx']=list(i)





    vectorizer2 = CountVectorizer(ngram_range=(1,1),vocabulary=train_features)

    namet = vectorizer2.fit_transform(test_data["name"])

    nameOneHott = pd.DataFrame(namet.toarray(),columns=vectorizer2.get_feature_names())

    df_test = pd.concat([test_data,nameOneHott],axis=1)



    #print(df_test)

    #define x and y

    x_label_test = df_test.keys().drop(['name','tag','tag-idx'])

    target_label_test = 'tag-idx'



    #predict

    test_pred = best.predict(df_test[x_label_test])



    #evaluate with real data

    from sklearn.metrics import accuracy_score

    score = accuracy_score(test_pred,df_test[target_label_test])

    print(bestn,'is  ',score)




