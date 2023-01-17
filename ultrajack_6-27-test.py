# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
mylist = [[6.1, 1.5], [18.6, 1.6], [2.3, 0.5], [2.6, 0.4]]
men_means[0]=(mylist[0][0])
men_means[0]=tuple(mylist[0][0])

women_means[0]=mylist[1][0]

men_means[1]=mylist[2][0]

women_means[1]=mylist[3][0]
N = 2





men_means = (mylist[0][0], mylist[2][0])

men_std = (2, 3)



ind = np.arange(N)  # the x locations for the groups

width = 0.35       # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(ind, men_means, width, color='r', alpha=0.3, yerr=men_std)



women_means = (mylist[1][0], mylist[3][0])

women_std = (3, 5)

rects2 = ax.bar(ind + width, women_means, width, color='b', alpha=0.3, yerr=women_std)



# add some text for labels, title and axes ticks

ax.set_ylabel('RTT')

ax.set_xticks(ind + width / 2)

#ax.grid(True)

ax.set_xticklabels(('50-Nodes', '90-Nodes'))

ax.legend((rects1[0], rects2[0]), ('Geo-Fw', 'Cross-Layer'))



plt.show()

data=pd.read_csv("../input/all.csv",index_col=None )
age_label=data.loc[:,['age']]

type_label=data.loc[:,['type']]

content_of_poem=data.loc[:,['content']]
data.iloc[318]
data=data[data.content.str.contains("Copyright")==False]
data[data.content.isnull()==True].index.tolist()
data.content.ix[data.content.str.contains("from Dana, 1904")==True]
data=data[data.content.str.contains("from Dana, 1904")==False]
data.content.str.lower()
data.content=data.content.str.replace('\n', " ")

data.content=data.content.str.replace("\t", " ")

data.content=data.content.str.replace("\r", " ")

data.content=data.content.str.replace(","," ").replace("."," ")

from nltk.corpus import stopwords

stop = stopwords.words('english')
#data.loc[:, 'content']=data.content.apply(lambda x: [item for item in x if item not in stop])

#data.content
remove_list=["A",

"An",

"The",

"Aboard",

"About",

"Above",

"Absent",

"Across",

"After",

"Against",

"Along",

"Alongside",

"Amid",

"Among",

"Amongst",

"Anti",

"Around",

"As",

"At",

"Before",

"Behind",

"Below",

"Beneath",

"Beside",

"Besides",

"Between",

"Beyond",

"But",

"By",

"Circa",

"Concerning",

"Considering",

"Despite",

"Down",

"During",

"Except",

"Excepting",

"Excluding",

"Failing",

"Following",

"For",

"From",

"Given",

"In",

"Inside",

"Into",

"Like",

"Minus",

"Near",

"Of",

"Off",

"On",

"Onto",

"Opposite",

"Outside",

"Over",

"Past",

"Per",

"Plus",

"Regarding",

"Round",

"Save",

"Since",

"Than",

"Through",

"To",

"Toward",

"Towards",

"Under",

"Underneath",

"Unlike",

"Until",

"Up",

"Upon",

"Versus",

"Via",

"With",

"Within",

"Without"]
#data.content=data.content.apply(lambda x: [item for item in x if item not in remove_list])

#data.content

for  value in remove_list:

    data.content=data.content.str.replace(value," ")
import re

# regular expression, using stemming: try to replace tail of words like ies to y 
data.content = data.content.str.replace("ing( |$)", " ")

data.content = data.content.str.replace("[^a-zA-Z]", " ")

data.content = data.content.str.replace("ies( |$)", "y ")

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, analyzer= 'word')

#vectorizer =TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
from sklearn.model_selection import train_test_split
#80 % training 20 % testing 

data_content_train,data_content_test, data_train_label,data_test_label =train_test_split(data.content,data.type,test_size = 0.2, random_state = 1)

null_list=data_content_train.index[data_content_train.isnull()==True].tolist()
data_content_train[data_content_train.isnull()==True].index
#data_content_train.dropna().index

data_content_test.isnull().sum()
train_ = vectorizer.fit_transform(data_content_train.tolist())

feature_names =vectorizer.get_feature_names()

feature_names

test_ = vectorizer.transform(data_content_test.tolist())
train_2 = vectorizer_one.fit_transform(data_content_train.tolist())

feature_names2 =vectorizer_one.get_feature_names()

from numpy import array
from sklearn.feature_selection import SelectKBest ,chi2

#y = np.array(data_content_train)

ch2 = SelectKBest(chi2, k=2000)

X_train=ch2.fit_transform(train_, data_train_label.tolist() )

X_test = ch2.transform(test_)

NAME2=np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
NAME2
import scipy.sparse as sp

if(sp.issparse(X_train)==True):

    X_train_dense = X_train.todense()

    X_test_dense = X_test.todense()

    
import xgboost as xgb
xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.6,

    'colsample_bytree': 1,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

a=le.fit_transform(data_train_label.tolist())

#le.transform(data_train_label.tolist()) 

list(le.inverse_transform(a))
dtrain = xgb.DMatrix(X_train, a )

dtest = xgb.DMatrix(X_test)
num_boost_rounds = 422

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

result =model.predict(dtest)

#test = vectorizer.transform(data["age"])

presult=pd.DataFrame(result)
presult
presult[(presult.values >= 0.5) & (presult.values < 1.5) ]= 1;

presult[(presult.values >= 1.5) & (presult.values < 2.5) ]=2;

presult[(presult.values >= -0.5) & (presult.values < 0.5) ]=0;
presult=presult.astype(int)


result_back=list(le.inverse_transform(presult))
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(data_test_label, result_back)

accuracy
data_test_label
result_back_s=(result_back.sort)

np.unique(result_back)