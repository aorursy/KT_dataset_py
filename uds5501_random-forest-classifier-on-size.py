# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
myDf = pd.read_csv('../input/AppleStore.csv', index_col = 'id')
myDf.drop(columns = ['Unnamed: 0'], inplace=True)
myDf.head()
# A utility to convert bytes to megabyte
def byteToMB(b):
    MB = b/(1024.0*1024)
    return MB

    
myDf['size_in_mb'] = myDf['size_bytes'].apply(byteToMB) 
myDf.drop(columns=['size_bytes'], inplace=True)
# Updated Dataframe
myDf.head()
myDf['prime_genre'].unique()
for i in myDf['prime_genre'].unique():
    newVar = myDf[myDf['prime_genre'] == i]
    newVar.sort_values(by = ['user_rating'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (newVar['track_name'][::-1][:6])
    print("\n")
    
for i in myDf['prime_genre'].unique():
    refinedDf = myDf[(myDf['rating_count_tot'] > 50000) & (myDf['prime_genre'] == i)]
    refinedDf.sort_values(['user_rating','rating_count_tot'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (refinedDf['track_name'][::-1][:6])
    print ("\n")
    
for i in myDf['prime_genre'].unique():
    refinedDf = myDf[(myDf['rating_count_tot'] > 20000) & (myDf['prime_genre'] == i) & (myDf['price'] == 0.00)]
    refinedDf.sort_values(['user_rating','rating_count_tot'], inplace = True)
    print("Top 5 for {} genre are".format(i))
    print (refinedDf['track_name'][::-1][:6])
    print ("\n")
    
eda2df = myDf[myDf['price'] == 0.00]
eda2df.sort_values(by = ['sup_devices.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num']][::-1].head(10)
eda2df = myDf[myDf['price'] == 0.00]
eda2df.sort_values(by = ['lang.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num']][::-1].head(10)
eda2df = myDf[myDf['price'] != 0.00]
eda2df.sort_values(by = ['sup_devices.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num', 'price']][::-1].head(10)
eda2df = myDf[myDf['price'] != 0.00]
eda2df.sort_values(by = ['lang.num'], inplace = True)
eda2df[['track_name', 'user_rating', 'size_in_mb', 'sup_devices.num', 'lang.num', 'price']][::-1].head(10)
numCol = myDf[['rating_count_tot', 'user_rating', 'sup_devices.num', 'price', 'lang.num', 'prime_genre']]
sns.pairplot(data = numCol, dropna=True, hue='prime_genre',palette='Set1')

sns.set_style("darkgrid")

plt.hist(myDf['price'], bins = 100)
# A utility function to create categories according to views
def df_categorizer(rating):
    if rating >= 100000:
        return 2
    elif rating < 10000:
        return 0
    else:
        return 1
myDf['pop_categories'] = myDf['rating_count_tot'].apply(df_categorizer)

finalDf = myDf[['size_in_mb', 'rating_count_tot', 'pop_categories']]
finalDf.groupby(['pop_categories']).mean()
# Importing the tasty stuff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
X = finalDf['size_in_mb']
y = finalDf['pop_categories']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 21)
npX_train = np.array(X_train)
npX_train = npX_train.reshape(-1,1)

npX_test = np.array(X_test)
npX_test = npX_test.reshape(-1,1)
scaler = StandardScaler()

npX_train = scaler.fit_transform(npX_train)
npX_test = scaler.transform(npX_test)
classifier = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=42)
classifier.fit(npX_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(npX_test)

#Reverse factorize (converting y_pred from 0s,1s and 2s to poor, average and popular
reversefactor = dict(zip(range(3),['poor', 'average', 'popular']))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)

# Making the Confusion Matrix
cnf_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual Species'], colnames=['Predicted Species'])
cnf_matrix
