# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

np.random.RandomState(seed=1)

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/adult-training.csv",names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])

test=pd.read_csv("../input/adult-test.csv",names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary'])

# sns.pairplot(train,hue='salary')
train['salary'] = train['salary'].apply(lambda x: 1 if x==' >50K' else 0)

test['salary'] = test['salary'].apply(lambda x: 1 if x==' >50K.' else 0)

train_y=train['salary']

test_y=test['salary']

train=train.drop(['salary'],axis=1)

test=test.drop(['salary'],axis=1)
train.head()
test=test.drop([0])

test_y=test_y.drop([0])

test.head(10)
train.info()

train['age']=train['age'].apply(int)

test['age']=test['age'].apply(int)
def regionize(country):

    if country in [' Holand-Netherlands',' Hungary',' Scotland',' Portugal',' Ireland',' Greece',' Poland',' France',' Italy',' England',' Germany',' Yugoslavia',' South']:

        return "Europe"

    elif country in [' Laos',' Thailand',' Vietnam',' Hong',' Iran',' Taiwan',' China',' Japan',' India',' Philippines']:

        return "Asia"

    elif country in [' Columbia',' Honduras',' Trinadad&Tobago',' Dominican-Republic',' Nicaragua',' Peru',' Guatemala',' Haiti',' Ecuador',' Cambodia',' El-Salvador',' Jamaica',' Puerto-Rico',' Cuba',' Mexico']:

        return "Latin"

    else:

        return "US"

    

def regionize2(country):

    if country in [' United-States', ' Cuba', ' ?']:

        return 'US'

    elif country in [' England', ' Germany', ' Canada', ' Italy', ' France', ' Greece', ' Philippines']:

        return 'Western'

    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', ' Columbia', ' Laos', ' Portugal', ' Haiti',

                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 

                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua', ' Vietnam', ' Holand-Netherlands' ]:

        return 'Poor' # no offence

    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', ' Japan', ' Yugoslavia', ' China', ' Hong']:

        return 'Eastern'

    elif country in [' South', ' Poland', ' Ireland', ' Hungary', ' Scotland', ' Thailand', ' Ecuador']:

        return 'Poland team'

    

    else: 

        return country

def draw_country_stacked_bars(data,title,size=(20,20)):

    data=data.sort_values(by=1)

    f, ax = plt.subplots(figsize=size)



    ind = np.arange(len(data.index))

    p1 = plt.barh(ind, data[1].values,color='r')

    p2 = plt.barh(ind, data[0],left=data[1],alpha=0.3)



    plt.yticks(ind, data.index)

    plt.xticks(np.arange(0, 110, 5))

    plt.legend((p1[0], p2[0]), ('>50K', '=<50K'))

    plt.title(title)

    plt.show()
t_y=pd.concat([test_y,train_y])

alldata=pd.concat([test,train])

alldata=pd.concat([alldata,t_y],axis=1)

sns.countplot(alldata['salary'])
f, ax = plt.subplots(figsize=(40, 20))

temp=alldata[~alldata['native-country'].isin([alldata['native-country'][0]])]

result = temp.groupby(["native-country"]).count().reset_index().sort_values('salary',ascending=False)

sns.countplot(x='native-country',data=temp,order=result['native-country'])


w0=alldata.groupby(['native-country','salary'])['salary'].count()

sums=w0.groupby(w0.index.to_frame()['native-country']).sum()

w0/=sums/100

w0=w0.unstack().fillna(0).sort_values(by=[1])

draw_country_stacked_bars(w0,"Procentowy rozkład zarobków według państw")

alldata['Region']=alldata['native-country'].apply(regionize)

alldata['Region2']=alldata['native-country'].apply(regionize2)
w1=alldata.groupby(['Region','salary'])['salary'].count()

sums=w1.groupby(w1.index.to_frame().Region).sum()

w1/=sums/100

w1=w1.unstack()

draw_country_stacked_bars(w1,"Procentowy rozkład zarobków według regionu urodzenia",(10,10))
w2=alldata.groupby(['Region2','salary'])['salary'].count()

sums=w2.groupby(w2.index.to_frame().Region2).sum()

w2/=sums/100

w2=w2.unstack()

draw_country_stacked_bars(w2,"Procentowy rozkład zarobków według innego regionu urodzenia",(10,10))
result = alldata.groupby(["occupation"]).count().reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(30, 10))

sns.countplot(alldata['occupation'],order=result['occupation'])
result = alldata.groupby(["occupation"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(30, 10))

sns.barplot(alldata['occupation'],alldata['salary'],order=result['occupation'])

f, ax = plt.subplots(figsize=(50, 10))

sns.distplot(alldata['age'])
f, ax = plt.subplots(figsize=(50, 10))

sns.barplot(alldata['age'],alldata['salary'])
def aproximate_age(age):

    if age<30:

        return "Young"

    elif age <60:

        return "Mid"

    else:

        return "Old"
alldata['approximated_age']=alldata['age'].apply(aproximate_age)
result = alldata.groupby(["approximated_age"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['approximated_age'],alldata['salary'],order=result['approximated_age'])
result = alldata.groupby(["workclass"]).count().reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(alldata['workclass'],order=result['workclass'])
result = alldata.groupby(["workclass"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['workclass'],alldata['salary'],order=result['workclass'])
f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(alldata['sex'])
f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['sex'],alldata['salary'])
result = alldata.groupby(["race"]).count().reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(alldata['race'],order=result['race'])
result = alldata.groupby(["race"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['race'],alldata['salary'],order=result['race'])
f, ax = plt.subplots(figsize=(50, 10))

sns.countplot(alldata['hours-per-week'])
f, ax = plt.subplots(figsize=(50, 10))

sns.barplot(alldata['hours-per-week'],alldata['salary'])
def aproximate_hours(hours):

    if hours<35:

        return "Low"

    elif hours<61:

        return "Mid"

    else:

        return "High"
alldata['aproximated_hours-per-week']=alldata['hours-per-week'].apply(aproximate_hours)
result = alldata.groupby(["aproximated_hours-per-week"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['aproximated_hours-per-week'],alldata['salary'],order=result['aproximated_hours-per-week'])
result = alldata.groupby(["education"]).count().reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.countplot(alldata['education'],order=result['education'])
result = alldata.groupby(["education"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['education'],alldata['salary'],order=result['education'])
def aproximate_edu(edu):

    if edu<9:

        return "Low"

    elif edu<13:

        return "Mid"

    else:

        return "High"
alldata['aproximated_education-num']=alldata['education-num'].apply(aproximate_edu)
result = alldata.groupby(["aproximated_education-num"])['salary'].aggregate(np.mean).reset_index().sort_values('salary',ascending=False)

f, ax = plt.subplots(figsize=(15, 10))

sns.barplot(alldata['aproximated_education-num'],alldata['salary'],order=result['aproximated_education-num'])
corr=alldata.corr()

sns.heatmap(corr)
alldata=alldata.drop(['fnlwgt'],axis=1)
alldata.isnull().sum()
num_feat=alldata.dtypes[alldata.dtypes!="object"].index

skews_col=alldata[num_feat].skew().sort_values(ascending=False)

skews_col = skews_col[abs(skews_col) > 0.5]

skews_col
from scipy.special import boxcox1p



for c in skews_col.index:

    if c not in ['salary']:

        alldata[c]=boxcox1p(alldata[c],0.15)
from sklearn.preprocessing import LabelEncoder



lab_cols=['aproximated_education-num']



for col in lab_cols:   

    lbl = LabelEncoder() 

    lbl.fit(alldata[col]) 

    alldata[col] = lbl.transform(alldata[col])



# lbl = LabelEncoder() 

# lbl.fit(list(test_y.values)) 

# test_y = lbl.transform(list(test_y.values))
# sns.pairplot(alldata,hue='salary')
alldata=alldata.drop(['salary'],axis=1)

alldata=alldata.drop(['Region'],axis=1)
alldata=pd.get_dummies(alldata)

alldata.head()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

sc.fit(alldata)

alldata=sc.transform(alldata)





train=alldata[len(test):]

test=alldata[:len(test)]
from sklearn.linear_model import LogisticRegression



from sklearn.ensemble import GradientBoostingClassifier





m=LogisticRegression()



m.fit(train,train_y)



m.score(test,test_y)

predictions=m.predict(test)



pop=np.asarray([1 if predictions[i]==test_y.values[i] else 0 for i in range(len(predictions)) ]).sum()/len(test_y)

pop
def sigmoid(x):

    return 1 / (1 + np.exp(-x))
np.random.RandomState(seed=1)

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



class OwnLogisticRegression(BaseEstimator, TransformerMixin, RegressorMixin):

    

    def __init__(self,lr,iterations):

        self.lr=lr

        self.iterations=iterations

        

    def fit(self,X,y):

        self.weights=np.zeros(X.shape[1])

        for ite in range(self.iterations):

            y_p=self.predict(X)

            grad=np.dot(X.T,y_p-y)              

            grad/=len(X)

            grad*=self.lr

            self.weights-=grad     

        return self

    

    def predict(self,X):

        return sigmoid(np.dot(X,self.weights))

    

    def cost(self,y_p,y):

        return -(1/y.shape[0])*np.sum(y*np.log(y_p)+(1-y)*np.log(1-y_p))



reg=OwnLogisticRegression(0.2,1000)

%time reg.fit(train,train_y)

predictions=reg.predict(test)

predictions=[1 if p>0.5 else 0 for p in predictions]

pop=np.asarray([1 if predictions[i]==test_y.values[i] else 0 for i in range(len(predictions)) ]).sum()/len(test_y)

pop