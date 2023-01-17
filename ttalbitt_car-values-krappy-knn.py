import pandas
from sklearn.model_selection import train_test_split


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    
    files = []
    
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
# I don't feel like dealing with the NaNs in the unclean files
# I want a 'pure' kNN

clean_files = files[:3]
clean_files.append(files[4])
clean_files += files[6:]
# weak kNN attempt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import sklearn.metrics
import numpy

for f in clean_files:
    
        data = open(f, 'r')
        df = pandas.read_csv(data)
        print(df['tax'], df['price'])
        df = pandas.get_dummies(df)
        X = df.drop('price', axis=1)
        y = df['price']
        X = normalize(X)
        
        kNN = KNeighborsClassifier(n_neighbors=10)
        kNN.fit(X, y)
        predictions = []
        k_nearest = kNN.kneighbors(X)
        
        preds = []
        for indices in k_nearest[1]:
            prices = []
            for index in indices[1:]:
                price = y[index]
                prices.append((9 - len(prices)) * price)
            preds.append(sum(prices)/45)  # prices' importance weighted by proximity
        
        print(f[-10:], sklearn.metrics.r2_score(y, preds))  # some are decent (.65ish), and others suck
        print(sum(abs(y - numpy.array(preds)))/len(preds))
# LinReg attempt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import sklearn.metrics
from sklearn.model_selection import train_test_split

def change_model(model, vc, n):
    if vc[model] < .01 * n:
        return ' Other'
    else:
        return model

avg_misses = []
for f in clean_files:
    
        data = open(f, 'r')
        df = pandas.read_csv(data)
        
        #df['model'] = df['model'].apply(change_model, args=( df['model'].value_counts(), df.shape[0]))  # get rid of infrequent models (eventual feature reduction)

        df = pandas.get_dummies(df)
        try:
            X = df.drop(['price'], axis=1)
        except:
            X = df.drop('price', axis=1)
        y = df['price']
        X = normalize(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
        
        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        
        preds = lin_reg.predict(X_test)
        print(f[-10:], sklearn.metrics.r2_score(y_test, preds))  # some r2 values are whack (e.g. ford w/ -3.7 r2)
        avg_misses.append(sum(abs(y_test - preds))/preds.shape[0])
print('costish:', sum(avg_misses)/len(avg_misses))
# another kNN attempt, where I throw out some of the models before getting dummies (making a lot of one hot columns)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
import sklearn.metrics
import numpy

def change_model(model, vc, n):
    if vc[model] < .01 * n:
        return ' Other'
    else:
        return model

for f in clean_files:
    
        data = open(f, 'r')
        df = pandas.read_csv(data)
        
        df['model'] = df['model'].apply(change_model, args=( df['model'].value_counts(), df.shape[0]))  # get rid of infrequent models (eventual feature reduction)
        
        df = pandas.get_dummies(df)
        X = df.drop('price', axis=1)
        y = df['price']
        X = normalize(X)
                
        kNN = KNeighborsClassifier(n_neighbors=10)
        kNN.fit(X, y)
        predictions = []
        k_nearest = kNN.kneighbors(X)
        
        preds = []
        for indices in k_nearest[1]:
            prices = []
            for index in indices[1:]:
                price = y[index]
                prices.append((9 - len(prices)) * price)
            preds.append(sum(prices)/45)  # prices' importance weighted by proximity
        
        print(f[-10:], sklearn.metrics.r2_score(y, preds))  # some are decent (.65ish), and others suck
        print(sklearn.metrics.explained_variance_score(y, preds))
        print(sum(abs(y - numpy.array(preds)))/len(preds))
