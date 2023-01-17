# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
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

vectorizer_one =TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
data.head()
data[["content","author","poem name"]]
from sklearn.model_selection import train_test_split
data.dropna(inplace=True)
#80 % training 20 % testing for label motion

data_content_train,data_content_test, data_train_label,data_test_label =train_test_split(data[["content","author","poem name"]],data.type,test_size = 0.2, random_state = 1)

data_test_label_for_age=data.ix[data_test_label.index].age

data_train_label_for_age=data.ix[data_train_label.index].age
data_train_label.size
#data_content_train.dropna().index

data_content_test.isnull().sum()
data_content_train
train_ = vectorizer.fit_transform(data_content_train.content.as_matrix())

feature_names =vectorizer.get_feature_names()

feature_names

test_ = vectorizer.transform(data_content_test.content.as_matrix())
train_2 = vectorizer_one.fit_transform(data_content_train.content.as_matrix())

feature_names2 =vectorizer_one.get_feature_names()
removelist=data_content_train["poem name"].index[data_content_train["poem name"].isnull()==True].tolist()

removelist
from sklearn import preprocessing

label_au = preprocessing.LabelEncoder()

label_author=label_au.fit_transform(data_content_train.author.as_matrix())

label_authorT=label_au.fit_transform(data_content_test.author.as_matrix())



label_poe_name =TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')  

label_poena=label_poe_name.fit_transform(data_content_train["poem name"].as_matrix())

label_poenaT  =label_poe_name.fit_transform(data_content_test["poem name"].as_matrix())



train_.shape
label_author=np.reshape(label_author, (label_author.shape[0], 1))

label_authorT=np.reshape(label_authorT, (label_authorT.shape[0], 1))
# Question: trying to combnine feature

# train_ with other features 

# like label_author,label_poena

#

#train_=sp.hstack(train_,label_author, label_poena)

#test_=sp.hstack(test_,label_authorT,label_poenaT)
train_.shape # 

test_.shape #  
from numpy import array
from sklearn.feature_selection import SelectKBest ,chi2

# find out that using SelectKBest did not improve the

# accuracy of the result

#

#

#y = np.array(data_content_train)

ch2 = SelectKBest(chi2, k=2000)

#X_train=ch2.fit_transform(train_, data_train_label.tolist() )

#X_test = ch2.transform(test_)

X_train=train_;

X_test=test_;

#NAME2=np.asarray(vectorizer.get_feature_names())[ch2.get_support()]
import scipy.sparse as sp

#if(sp.issparse(X_train)==True):

#   X_train = X_train.todense()

#   X_test = X_test.todense()

    
import xgboost as xgb

xgb_params = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.6,

    'colsample_bytree': 1,

    'objective': 'reg:linear',

    "eval_metric": 'logloss',

    'silent': 1

}
xgb_params_age = {

    'eta': 0.05,

    'max_depth': 6,

    'subsample': 0.6,

    'colsample_bytree': 1,

    'objective': 'reg:linear',

    "eval_metric": 'error',

    'silent': 1

}
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

a=le.fit_transform(data_train_label.as_matrix())

type(a)
le2 = preprocessing.LabelEncoder()

a_age=le2.fit_transform(data_train_label_for_age.as_matrix())

type(a_age)
dtrain = xgb.DMatrix(X_train, a )

dtest = xgb.DMatrix(X_test)

dtrain_age = xgb.DMatrix(X_train, a_age )

dtest_age = xgb.DMatrix(X_test)
num_boost_rounds = 422

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

result =model.predict(dtest)

#test = vectorizer.transform(data["age"])

num_boost_rounds = 422

model_age = xgb.train(dict(xgb_params_age, silent=0), dtrain_age, num_boost_round=num_boost_rounds)

result_age =model.predict(dtest_age)

#test = vectorizer.transform(data["age"])
result_age
data_test_label_for_age
presult=pd.DataFrame(result)

presult_age=pd.DataFrame(result_age)
presult[(presult.values >= 0.5) & (presult.values < 1.5) ]= 1;

presult[(presult.values >= 1.5) & (presult.values < 2.5) ]=2;

presult[(presult.values >= -0.5) & (presult.values < 0.5) ]=0;



presult_age[(presult_age.values >= -0.5) & (presult_age.values < 0.5) ]=0;

presult_age[(presult_age.values >= 0.5) & (presult_age.values < 1.5) ]= 1;
presult=presult.astype(int)

presult_age=presult_age.astype(int)
np.unique(presult) #make sure results have three categories


result_back=le.inverse_transform(presult.values)

result_back_age=le2.inverse_transform(presult_age.values)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(data_test_label, result_back)

accuracy
accuracy_age = accuracy_score(data_test_label_for_age, result_back_age)

accuracy_age
pd.DataFrame({  'poem name': data_content_test["poem name"],

                'correct_data' : data_test_label_for_age+ " " +data_test_label,

                'predict' : result_back_age.ravel()+" " +result_back.ravel()

                    })
result_back_s=(result_back.sort)

np.unique(result_back)