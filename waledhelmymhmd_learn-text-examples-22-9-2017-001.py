# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) 

#will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



dp = pd.read_csv('../input/spam.csv', encoding='latin-1')

dp.head()
dp = dp.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

dp = dp.rename(columns = {'v1':'Class','v2':'Text'})

dp.head()
# Count the number of words in each Text

dp['Count']=0 # عمل حقل جديد

for i in np.arange(0,len(dp.Text)):

    dp.loc[i,'Count'] = len(dp.loc[i,'Text'])



# Unique values in target set

print ("Unique values in the Class set: ", dp.Class.unique())



# Replace ham with 0 and spam with 1

dp = dp.replace(['ham','spam'],[0, 1]) 



# displaying the new table

dp.head()
dp.loc[0,'Count'] # لمعرفة البيانات المسجلة في حقل معين
dp.loc[0, 'Text']
len(dp.loc[0, 'Text']) # لمعرفة عدد الكلمات
# تحميل قاعدة بيانات موجودة في المكتبة

from sklearn.datasets import load_iris

iris = load_iris()



# تخزين خصائص البيانات (المدخلات) في المصفقوفة 

# X

# و الاستجابة(المخرجات) في المتغير 

# y

X = iris.data

y = iris.target



# لاحظ شكل 

# X and y

print(X.shape) # مصفوفة ثنائية

print(y.shape)

import pandas as pd

# فحص اول 5 سطور من مصفوفة المدخلات

pd.DataFrame(X, columns = iris.feature_names).head()
y # المخرجات

# 0 or 1 or 2
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()



# ادخال البيانات للنموذج



knn.fit(X, y)
# لاتمام عملية التوقع بطريقة سليمة لابد من ان يكون مدخل عملية التوقع له نفس عدد الخصائص للمدخلات

knn.predict([[3, 5, 4, 2]])
from sklearn.feature_extraction.text import CountVectorizer

simple_train = ['call you tonight','call me a cab', 'please call me.... PLEASE']

vect = CountVectorizer()

vect.fit(simple_train)

vect.get_feature_names()

simple_train_dtm = vect.transform(simple_train)

simple_train_dtm
simple_train_dtm.toarray()
pd.DataFrame(simple_train_dtm.toarray(), columns = vect.get_feature_names() )
type(simple_train_dtm)
print(simple_train_dtm)
# Test the model

simple_test = ["please don't call me"]

simple_test_dtm = vect.transform(simple_test)

simple_test_dtm.toarray()

pd.DataFrame(simple_test_dtm.toarray(), columns = vect.get_feature_names() )
sms = pd.read_csv('../input/spam.csv', encoding='latin-1')

sms.shape
sms.head(10)
sms = sms.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)

sms = sms.rename(columns = {'v1':'lable','v2':'message'})

sms.head()
sms.lable.value_counts()
# Replace ham with 0 and spam with 1

#sms = dp.replace(['ham','spam'],[0, 1])

sms['lable_num'] = sms.lable.map({'ham':0, 'spam':1})

sms.head(10)
X = sms.message

y = sms.lable_num

print(X.shape) # المدخلات لا بد ان تكون مصفوفة ثنائية و هي الان اوحادية لا بد من تحويلها

print(y.shape)
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
vect = CountVectorizer()

vect.fit(X_train)

X_train_dtm = vect.transform(X_train)

#vect.get_feature_names()

X_train_dtm = vect.fit_transform(X_train)

X_train_dtm
X_test_dtm = vect.transform(X_test)

X_test_dtm
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
%time nb.fit(X_train_dtm, y_train)
y_pred_class = nb.predict(X_test_dtm)

from sklearn import metrics

metrics.accuracy_score(y_test, y_pred_class)
metrics.confusion_matrix(y_test, y_pred_class)
# طباعة مقطع الرسالة ل

# ham incorrect classifier

X_test[(y_pred_class==1)&(y_test==0)]
# طباعة مقطع الرسالة ل

# spam incorrect classifier

X_test[(y_pred_class==0)&(y_test==1)]
# طباعة مقطع الرسالة ل

# spam incorrect classifier

# طريقة اخرى لمعرفة الخطا

X_test[y_pred_class< y_test]
# example false negative

X_test[4674]
# طباعة احتمالية ظهور الرقم 0 

nb.predict_proba(X_test_dtm)[:, 0]
# طباعة احتمالية ظهور الرقم 1

y_pred_prob = nb.predict_proba(X_test_dtm)[:, 1]

y_pred_prob
# طباعة دقة النموذج

metrics.roc_auc_score(y_test, y_pred_prob)
# تخزين الكلمات في متغير

X_train_tokens = vect.get_feature_names()

len(X_train_tokens)

# طباعة اول 50 كلمة

print(X_train_tokens[0:50])
# طباعة اخر 50 كلمة

print(X_train_tokens[-50:])
# يقوم النموذج بحساب عدد المرات التي تظهر بها الكلمة في كل نوع من المخرجات 

# spam or ham

nb.feature_count_
# الاعمدة تمثل كلمات المدخلات و الصفوف تمثل نوع المخرجات 

nb.feature_count_.shape
# طريقة حساب عدد الكلمات الموجودة في المدخلات في المخرج 

# ham

ham_token_count = nb.feature_count_[0, :]

ham_token_count
# طريقة حساب عدد الكلمات الموجودة في المدخلات في المخرج 

# spam

spam_token_count = nb.feature_count_[1, :]

spam_token_count
# وضع النتائج السابقة في جدول

tokens = pd.DataFrame({'token' :X_train_tokens , 'ham': ham_token_count , 

                      'spam':  spam_token_count})

tokens.head()
tokens.sample(5, random_state=7)
nb.class_count_
# اضافة رقم 1 الى عمودي 

# ham and spam

tokens['ham'] = tokens.ham + 1

tokens['spam'] = tokens.spam + 1

tokens.sample(5, random_state=7)
# قياس نسبة كل من

# ham and spam

# الى نسبة ظهورهم في المخرجات

tokens['ham'] = tokens.ham / nb.class_count_[0]

tokens['spam'] = tokens.spam / nb.class_count_[1]

tokens.sample(5, random_state=7)
# حساب نسبة 

# spam to ham

# لكل كلمة في المدخلات

tokens['spam_ratio'] = tokens.spam / tokens.ham

tokens.sample(5, random_state=7)
# ترتيب النسبة من الاعلى الى الاصغر

tokens.sort_values('spam_ratio', ascending=False)
# البحث نسبة المهمل لكلمة معينة

tokens.loc[1768, 'spam_ratio']