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

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/clinvar_conflicting.csv')
#df = df.dropna()
#df.info()
fig = plt.figure(figsize = (10, 10))
sns.countplot(x= 'CLASS', data = df, hue = 'CHROM', palette='icefire')
fig = plt.figure(figsize = (10, 10))
sns.heatmap(df.isnull(), cmap = 'viridis', cbar = False)
toBeConsidered = ['CHROM', 'POS', 'REF', 'ALT', 'AF_ESP', 'AF_EXAC', 'AF_TGP',
       'CLNDISDB', 'CLNDN', 'CLNHGVS', 'CLNVC','MC', 'ORIGIN', 'CLASS',
       'Allele', 'Consequence', 'IMPACT', 'SYMBOL', 'Feature_type',
       'Feature', 'BIOTYPE', 'STRAND','CADD_PHRED', 'CADD_RAW']
df2 = df[toBeConsidered]
df2 = df2.dropna()

cutdowns = []
for i in df2.columns.values:
    if df2[i].nunique() < 1000:
        cutdowns.append(i)
print("The selected Columns for training are : ", cutdowns)
df_final = df2[cutdowns]
#df_final.info()
df_final['CHROM'] = df_final['CHROM'].astype(str)
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed1 = fh.fit_transform(df_final['REF'])
hashed1 = hashed1.toarray()
hashedFeatures1 = pd.DataFrame(hashed1)
nameList = {}
for i in hashedFeatures1.columns.values:
    nameList[i] = "REF"+str(i+1)


hashedFeatures1.rename(columns = nameList, inplace = True)
print("The Hashed REF table is somethinng like this : \n",hashedFeatures1.head())
#df['ALT']
#fh = FeatureHasher(n_features = 5, input_type = 'string')
hashed2 = fh.fit_transform(df_final['ALT'])
hashed2 = hashed2.toarray()
hashedFeatures2 = pd.DataFrame(hashed2)

nameList2 = {}
for i in hashedFeatures2.columns.values:
    nameList2[i] = "ALT"+str(i+1)


hashedFeatures2.rename(columns = nameList2, inplace = True)
print("The Hashed ALT table is somethinng like this : \n",hashedFeatures2.head())
binaryFeature1 = pd.get_dummies(df_final['CLNVC'])
print("While the One hot encoded matrix of CLNVC Columns is like this : \n")
binaryFeature1.head()
df_final = df_final.drop(columns=['MC'], axis = 1)
hashed0 = fh.fit_transform(df_final['CHROM'])
hashed0 = hashed0.toarray()
hashedFeatures0 = pd.DataFrame(hashed0)

nameList0 = {}
for i in hashedFeatures0.columns.values:
    nameList0[i] = "CHROM"+str(i+1)


hashedFeatures0.rename(columns = nameList0, inplace = True)
hashedFeatures0.head()
hashed3 = fh.fit_transform(df_final['Allele'])
hashed3 = hashed3.toarray()
hashedFeatures3 = pd.DataFrame(hashed3)

nameList3 = {}
for i in hashedFeatures3.columns.values:
    nameList3[i] = "Allele"+str(i+1)


hashedFeatures3.rename(columns = nameList3, inplace = True)
hashedFeatures3.head()
hashed4 = fh.fit_transform(df_final['Consequence'])
hashed4 = hashed4.toarray()
hashedFeatures4 = pd.DataFrame(hashed4)

nameList4 = {}
for i in hashedFeatures4.columns.values:
    nameList4[i] = "Consequence"+str(i+1)


hashedFeatures4.rename(columns = nameList4, inplace = True)
hashedFeatures4.head()
df_final['IMPACT'].nunique()
binaryFeature3 = pd.get_dummies(df_final['IMPACT'])
binaryFeature3.head()
df_final = df_final.drop(columns=['Feature_type'], axis = 1)
binaryFeature4 = pd.get_dummies(df_final['BIOTYPE'], drop_first=True)
binaryFeature4.head()
binaryFeature5 = pd.get_dummies(df_final['STRAND'], drop_first=True)
binaryFeature5.head()
df3 = pd.concat([binaryFeature1, binaryFeature3, binaryFeature4, binaryFeature5, hashedFeatures1 , hashedFeatures2, hashedFeatures3, hashedFeatures4,hashedFeatures0, df_final['CLASS']], axis=1)
df3 = df3.dropna()
df3.rename(columns={1 : "one", 16 : "sixteen"}, inplace = True)
print(df3.columns.values)
df3.head()
y = df3['CLASS']
X = df3.drop(columns=['CLASS'], axis = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_lr))
dt = DecisionTreeClassifier(max_depth=6)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_dt))
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_rf))
gra = GradientBoostingClassifier()
gra.fit(X_train, y_train)
pred_gra = gra.predict(X_test)
print( "Classification Report :\n ", classification_report(y_test, pred_gra))
from collections import OrderedDict
feature_imp = {}
for i in zip(X.columns, lr.coef_[0]):
    feature_imp[i[0]] = i[1]
final_imp = OrderedDict(feature_imp)
df_features = pd.DataFrame(final_imp, index = range(1)).T
df_features.rename(columns={0: "Importance_lr"}, inplace = True)

my_colors = ['g', 'b']*5

df_features.plot(kind='bar',figsize = (20,5), color = my_colors)
#list(feature_imp.values())
feature_imp2 = {}
for i in zip(X.columns, rf.feature_importances_):
    feature_imp2[i[0]] = i[1]

final_imp2 = OrderedDict(feature_imp2)
#print(feature_imp2)
df_features2 = pd.DataFrame(final_imp2, index = range(1)).T
df_features2.rename(columns={0: "Importance_rf"}, inplace = True)
df_features2.plot(kind='bar',figsize = (15, 5), color = my_colors)
df_compare = pd.concat([df_features, df_features2], axis = 1)
df_compare.plot(kind='bar',figsize = (20, 5))
from keras.models import Sequential
from keras.layers import (Dense, Flatten, Dropout, BatchNormalization)
model = Sequential()
model.add(Dense(128 , input_dim = 38, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.33))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(X, y, batch_size=64, epochs = 20, verbose=1)
prediction = model.predict(X_test)
def finalPredictions(x):
    if x<0.5 : 
        return 0
    else:
        return 1
pred_deep = []
for i in prediction:
    pred_deep.append(finalPredictions(i))
    
print(classification_report(y_test, pred_deep))
