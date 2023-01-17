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
test = pd.read_csv("../input/test_cleaned.csv",encoding='latin-1')

train = pd.read_csv("../input/train.csv",encoding='latin-1')

train.head()  #regarder les 5 first
train.info() #on voit qu'il a des colonnes non nulles.
train.fillna(' ', inplace = True)

train['message'] = train['v2'] + train['Unnamed: 2'] + train['Unnamed: 3'] + train['Unnamed: 4']#combiner les colonnes non nulles pour ne pas perdre l'info

train.head()
train = train.drop(["v2", "Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1) #enlever colonnes inutiles

train = train.rename(columns={"v1":"Spam"})

train.head()
train = train.replace(['ham','spam'],[0, 1]) #remplacer par des 0 et 1

train.head()
#Convertir en matrices

y_train = train['Spam'].as_matrix()

x_train_text = train['message'].as_matrix()



#test data

x_TEST_text = test['message'].as_matrix()
from sklearn.feature_extraction.text import CountVectorizer



# Instantiate the CountVectorizer method

count_vector = CountVectorizer()



# Fit the training data and then return the matrix as sparse

training_data = count_vector.fit_transform(x_train_text)



testing_data = count_vector.transform(x_TEST_text) 
#Import and use multinomial

from sklearn.naive_bayes import MultinomialNB 



#The multinomial Naive Bayes classifier is suitable for classification with discrete features 

#(e.g., word counts for text classification).



naive_bayes = MultinomialNB()

naive_bayes.fit(training_data,y_train) #train our model
#Obtain our predictions using our model

predictionsM = naive_bayes.predict(testing_data)
#Preparing the submission file

test_submission = pd.DataFrame({'Spam': np.array(predictionsM).flatten()})  #add our predictions 

test_submission["Id"] = (test_submission.index + 1)  #add the Id for evaluation

test_submission= test_submission[['Id','Spam']]      #re-order for evaluation
#Produce the submission file as an output 

pd.DataFrame(test_submission).to_csv('SubmissionMulti.csv', index=False)
from sklearn.naive_bayes import GaussianNB



bayes_classifier = GaussianNB()

bayes_classifier.fit(training_data.toarray(), y_train) #need toarray() here
#Obtain predictions

predictionsB = bayes_classifier.predict(testing_data.toarray())
#preparing submission

test_submission2 = pd.DataFrame({'Spam': np.array(predictionsB).flatten()})

test_submission2["Id"] = (test_submission2.index + 1)

test_submission2= test_submission2[['Id','Spam']]
#Produce submission file in output

pd.DataFrame(test_submission2).to_csv('SubmissionG.csv', index=False)