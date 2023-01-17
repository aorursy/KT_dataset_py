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
df = pd.read_csv('../input/voice.csv')

df.head()
print("Total number of samples: {}".format(df.shape[0]))

print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
import seaborn as sns

# cross over of data is pretty similar

#sns.lmplot("meanfreq", "dfrange", hue="label", data=df, fit_reg=False)

sns.pairplot(df, hue="label")
#Test train split of 80/20 percent. 



from sklearn.cross_validation import train_test_split

X_train, X_test = train_test_split(df, test_size=0.2, random_state=0)

#Changes the gender into int as catagories

X_train['label_int']= X_train['label'].astype('category')

X_test['label_int']= X_test['label'].astype('category')

X_train = X_train.drop('label', axis=1)

X_test = X_test.drop('label', axis=1)

Y_train = X_train['label_int'].cat.codes 

Y_test =X_test['label_int'].cat.codes

X_train = X_train.drop('label_int', axis=1)

X_test = X_test.drop('label_int', axis=1)
#Applying Principle Component Analysis to remove the number of features. 



#from sklearn.decomposition import PCA

#pca = PCA(n_components=1)

#pca.fit(X_train)

#X_train_pca = pca.transform(X_train)

#X_test_pca = pca.transform(X_test)

#print(X_train.shape)

#print(X_test.shape)

#X_train.head()

#X_train_pca
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf.fit(X_train, Y_train)
prediction = clf.predict(X_test)
#Validation of between the prediction and actual data



from sklearn.metrics import accuracy_score

accuracy_score(Y_test, prediction)

#So many room for improvements!!!!