import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,confusion_matrix
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df_train = pd.read_csv("../input/train.csv");
df_test = pd.read_csv("../input/test.csv");
f,ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df_train.corr(), annot=True, fmt= '.3f',ax=ax)
df_train.shape

df_train.head(5)
df_train.describe()
pp = sns.pairplot(df_train, hue = 'popularity', size=1.8, diag_kind = 'kde')
pp.set(xticklabels=[])
train_size = df_train.shape[0] * 0.7;
X = df_train.loc[:train_size, df_train.columns != 'popularity']
Y = df_train.loc[:train_size, df_train.columns == 'popularity']
Y = np.ravel(Y)

XV = df_train.loc[train_size+1:, df_train.columns != 'popularity']
YV = df_train.loc[train_size+1:, df_train.columns == 'popularity']
YV = np.ravel(YV)
clf = tree.DecisionTreeClassifier(random_state=0)
#clf = RandomForestClassifier(random_state=0)
clf = clf.fit(X, Y);
ac = accuracy_score(YV,clf.predict(XV))
print('Accuracy is: ',ac)
cm = confusion_matrix(YV,clf.predict(XV))
sns.heatmap(cm,annot=True,fmt="d")
Y_pred = clf.predict(df_test)
print(Y_pred)