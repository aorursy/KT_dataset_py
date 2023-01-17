import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as skl
import math
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
%matplotlib inline
np.random.seed(42)
df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')
y_train = df_train['Satisfied']
train_custId = df_train['custId']
test_custId = df_test['custId']
df_train = df_train.drop(['Satisfied'], axis = 1)
df_train['TotalCharges'] = df_train['TotalCharges'].replace(" ","1")
df_train['TotalCharges'] = df_train['TotalCharges'].astype(float)
df_test['TotalCharges'] = df_test['TotalCharges'].replace(" ","1")
df_test['TotalCharges'] = df_test['TotalCharges'].astype(float)
#df_test = df_test.drop(['custId'], axis = 1)
binary_features = ['gender','Married','Children','Internet','AddedServices']
categorical_features = ['TVConnection','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','HighSpeed','Subscription','PaymentMethod']
binary_features = binary_features + categorical_features
imp_feat = 0
le = LabelEncoder()
for i in binary_features:
    df_train[i] = le.fit_transform(df_train[i])
    df_test[i] = le.transform(df_test[i])
#df_train = pd.get_dummies(df_train, columns = categorical_features,drop_first = True )
#df_test = pd.get_dummies(df_test, columns = categorical_features,drop_first = True )
n_clust = 3
#from sklearn.preprocessing import PowerTransformer
#scaler = PowerTransformer()
#df_train = pd.DataFrame(scaler.fit_transform(df_train.values), index=df_train.index, columns=df_train.columns)
#df_test = pd.DataFrame(scaler.transform(df_test.values), index=df_test.index, columns=df_test.columns)
standardScaler = StandardScaler()
df_train_scaled = pd.DataFrame(standardScaler.fit_transform(df_train), index=df_train.index, columns=df_train.columns)
df_test_scaled = pd.DataFrame(standardScaler.transform(df_test), index=df_test.index, columns=df_test.columns)
rand_State = 2
lda = LinearDiscriminantAnalysis()
lda_fit = lda.fit_transform(df_train_scaled, y_train)
new_test = pd.DataFrame(lda.transform(df_test_scaled))
df_test_scaled['lda_column'] = new_test[imp_feat]
new_train = pd.DataFrame(lda.transform(df_train_scaled))
df_train_scaled['lda_column'] = new_train[imp_feat]
df_train.head()
df_test.head()
max_auc = 0
max_clust = 0
for rand_state in range(27,87,5):
    for pp in range(1,15,1):
        kmeans = KMeans(n_clusters=pp,random_state=rand_state,n_jobs=-1)
        kmeans.fit(df_train_scaled)
        y_kmeans = kmeans.predict(df_train_scaled)
        kmeans_dict = {}
        for i in range(len(y_kmeans)):
            if y_kmeans[i] not in kmeans_dict:
                kmeans_dict[y_kmeans[i]] = [y_train[i]]
            else:
                kmeans_dict[y_kmeans[i]].append(y_train[i])
        map_dict={}        
        for i in kmeans_dict:
            map_dict[i] = max(set(kmeans_dict[i]), key=kmeans_dict[i].count)    
        for i in range(len(y_kmeans)):
            y_kmeans[i] = map_dict[y_kmeans[i]]
        y_pred = kmeans.predict(df_train_scaled)
        for i in range(len(y_pred)):
            y_pred[i] = map_dict[y_pred[i]]
        curr = roc_auc_score(y_train, y_pred)
        #print(curr,pp)
        if curr > max_auc:
            max_auc = curr
            max_clust = pp
kmeans = KMeans(n_clusters=n_clust,random_state=rand_State,n_jobs=-1)
kmeans.fit(df_train_scaled)
y_kmeans = kmeans.predict(df_train_scaled)
kmeans_dict = {}
for i in range(len(y_kmeans)):
    if y_kmeans[i] not in kmeans_dict:
        kmeans_dict[y_kmeans[i]] = [y_train[i]]
    else:
        kmeans_dict[y_kmeans[i]].append(y_train[i])
map_dict={}        
for i in kmeans_dict:
    map_dict[i] = max(set(kmeans_dict[i]), key=kmeans_dict[i].count)    
for i in range(len(y_kmeans)):
    y_kmeans[i] = map_dict[y_kmeans[i]]
y_pred = kmeans.predict(df_test_scaled)
for i in range(len(y_pred)):
    y_pred[i] = map_dict[y_pred[i]]
out = pd.concat([test_custId,pd.DataFrame(y_pred)],axis=1)
out.columns = ['custId','Satisfied']
out.to_csv('finalsub.csv', index=False)