#importing dataset and libraries

import numpy as np 

import pandas as pd 



df = pd.read_csv("/kaggle/input/star-dataset/6 class csv.csv")
df.shape
df.head()
#checking for missing values

df.isnull().sum()
#different star types and Spectral Classes

df['Star type'].value_counts() , df['Spectral Class'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
#checking correlation between variables for PCA

sns.heatmap(data = df.corr(), annot = True)
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()



df['Star_color'] = labelencoder.fit_transform(df['Star color'])

df['Spectral_Class'] = labelencoder.fit_transform(df['Spectral Class'])
features = df.drop(['Star type','Star color','Spectral Class'], axis = 1)

labels = df['Star type']
#scaling our training model

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaled_train_features = scaler.fit_transform(features)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(scaled_train_features)

exp_variance = pca.explained_variance_ratio_
fig, ax = plt.subplots()

ax.bar(range(pca.n_components_), exp_variance)

ax.set_xlabel('Principal Component number')
cum_exp_variance = np.cumsum(exp_variance)



fig, ax = plt.subplots()

ax.plot(cum_exp_variance)

ax.axhline(y=0.85, linestyle=':')
n_component = 2



pca = PCA(n_component, random_state=10)

pca.fit(scaled_train_features)

pca_projection = pca.transform(scaled_train_features)
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(pca_projection, labels, random_state=10)
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(random_state=10)

dt.fit(train_features, train_labels)

pred_labels_tree = dt.predict(test_features)
from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(random_state=10)

logreg.fit(train_features, train_labels)

pred_labels_logit = logreg.predict(test_features)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=10)

clf.fit(train_features, train_labels)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=10)



tree = DecisionTreeClassifier()

logreg = LogisticRegression()

clf = RandomForestClassifier()



tree_score = cross_val_score(tree, pca_projection, labels, cv=kf)

logit_score = cross_val_score(logreg, pca_projection, labels, cv=kf)

rt_score = cross_val_score(clf,pca_projection, labels, cv=kf)



# Mean of all the score arrays

print("Decision Tree:", np.mean(tree_score),"Logistic Regression:", np.mean(logit_score),"Random Forest:",np.mean(rt_score))