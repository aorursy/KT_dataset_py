# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#I am not implementing EDA in this task



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

heart_df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')

heart_df.describe()

heart=heart_df.drop(['DEATH_EVENT'],axis=1)

heart_out = heart_df['DEATH_EVENT']

#standardising the data

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

print(heart.mean())

print(heart.std())

scalar = StandardScaler()

heart_sd=scalar.fit_transform(heart)

print(heart_sd.std())# checking whether the standard deviation is 1 or not

heart_train = heart_sd[0:220:1, ::]

heart_train.shape

heart_test= heart_out[0:220:1]

heart_test
#spliting the data into train and cv 

#Whole data am only spliting into train and cv but no test 

heart_tr,heart_cv=train_test_split(heart_train, test_size=0.1,random_state=0)

heart_trts,heart_cvts = train_test_split(heart_test, test_size=0.1,random_state=0)

regg= LogisticRegression()

algo = regg.fit(heart_tr,heart_trts)

algo

herw=regg.predict(heart_cv)

print(herw)

score_tr=regg.score(heart_tr,heart_trts)

print(score_tr)

score_cv = regg.score(heart_cv,heart_cvts)

print(score_cv)



#here we can observe that training error is more and Cv error as well. This indicates there is something wrong.

# To diagnoise we plot learning curves to find what exactly went wrong(ex: high bias or high variance)
#in this block am finding the train error and validation error for different data sizes

from sklearn.model_selection import learning_curve

X=heart_train

y=heart_test

train_size=[70,100,120,150,197]

estimator=LogisticRegression()

train_sizes,train_scores,validation_scores=learning_curve(estimator,X,y,train_sizes=train_size,cv=12,error_score='raise',scoring = 'neg_mean_squared_error')

print(train_scores)

print(validation_scores)
#In this block we plot learning curves

train_scoremean = -train_scores.mean(axis=1)

validation_scoremean = -validation_scores.mean(axis=1)

print(train_scoremean)

print(validation_scoremean)

plt.style.use('seaborn')

plt.plot(train_sizes,train_scoremean,label='Training error')

plt.plot(train_sizes,validation_scoremean,label='validation error')

plt.ylabel('MSE',fontsize = 14)

plt.xlabel('training size')

plt.legend()

plt.title('Learning curves for logistic regression')

plt.ylim(0,0.4)





#By observing learning curve i found that the model has high bias. so in order to reduce high bias we add poly features to the model
#adding poly features to the model

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3,interaction_only=True)

poly_feat = poly.fit_transform(heart_sd)

scalarpoly = StandardScaler()

poly1 = scalarpoly.fit_transform(poly_feat)

poly_train,poly_cv = train_test_split(poly1,test_size=0.1,random_state=0)

poly_traintes,poly_cvtes = train_test_split(heart_out,test_size=0.1,random_state=0)

algo_poly = regg.fit(poly_train,poly_traintes)



algo_pre = regg.predict(poly_cv)

y_pre = regg.predict(poly_cv)



#print(algo_pre)

#poly_cvtes

score_t = regg.score(poly_train,poly_traintes)

print(score_t)

score_c = regg.score(poly_cv,poly_cvtes)

print(score_c)

                     
polytrain_size=[30,50,70,100,140,180,200]

train_sizes,train_scores,validation_scores= learning_curve(regg,X=poly_train,y=poly_traintes,train_sizes=polytrain_size,cv=12,scoring = 'neg_mean_squared_error')

train_polymean=-train_scores.mean(axis=1)

validation_polymean = -validation_scores.mean(axis=1)

#print(train_polymean)

#print(validation_polymean)

plt.plot(train_sizes,train_polymean,label='traning errors')

plt.plot(train_sizes,validation_polymean,label='cross validation errors')

plt.legend()

plt.ylabel('MSE')

plt.xlabel('training sizes')

plt.title('lc for poly regression')

plt.ylim(0,0.5)



#from the below graphs we can observe training error is zero and validation error is decreasing. But the problem is it has slight high variance. To reduce the variance we can likely to add more data which helps.