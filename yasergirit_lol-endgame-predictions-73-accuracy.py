# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.metrics import plot_roc_curve, accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# First look at data
url = "/kaggle/input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv"
data = pd.read_csv(url)
data.head()
# No missing data
data.info()
data.columns
data.columns.size
blue = data.iloc[:,2:-19]
blue.columns
red = data.iloc[:,21:]
red.columns
def cols_check(a,b):
    print(a.columns.size,b.columns.size)
    if red.columns.size == blue.columns.size:
        print("Number of columns equal")
    else:
        print("Not equal")
cols_check(blue,red)
#We have to make similar the column names to calculate between the two data sets. 
#For this reason, I will create a list and then that will be appended to new dataset column names.
temp_list = []
for i in range(19):
    temp_list.append(i)
    
temp_list
difference = blue.set_axis(temp_list, axis = 1) - red.set_axis(temp_list, axis = 1)
difference.head(15)
diff_cols = []

for i in temp_list:
    diff_cols.append(blue.columns.to_list()[i][4:])
    

diff_cols
difference.set_axis(diff_cols, axis = 1, inplace = True)
difference.head()
print(data[["blueGoldDiff","blueExperienceDiff"]],"\n\n", data[["redGoldDiff","redExperienceDiff"]])
difference.drop(columns = ["GoldDiff","ExperienceDiff"], inplace = True)
# We can also drop kills or deaths columns. I will drop deaths column.
difference.drop(columns = "Deaths", inplace = True)
difference.head()
# We can also drop first blood column
difference.drop(columns = "FirstBlood", inplace = True)
fig, ax = plt.subplots(figsize=(12,10)) 
sns.heatmap(difference.corr(), annot = True, linewidths=.3, cmap ="binary")
# GoldPerMin == TotalGol and CSPerMin == TotalMinionsKilled
difference.drop(columns = ["GoldPerMin","CSPerMin"], inplace = True)
#Train and Test Split

X = difference
y = data[["blueWins"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 7)
print("X_train shape is : ", X_train.shape)
print("X_test shape  is : ", X_test.shape)
print("y_train shape is : ", y_train.shape)
print("y_test shape is : ", y_test.shape)
clf = svm.SVC()
clf.fit(X_train,y_train.values.ravel())

y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

svc_disp = plot_roc_curve(clf, X_test, y_test)
clfs = SGDClassifier(loss="hinge", penalty="l2", max_iter=165)
clfs.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

sgd_disp = plot_roc_curve(clfs, X_test, y_test)
nc = KNeighborsClassifier(n_neighbors=17)
nc.fit(X_train, y_train.values.ravel())

y_pred = nc.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

knn_disp = plot_roc_curve(nc, X_test, y_test)
clf = LogisticRegression(random_state=5, max_iter = 1000).fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

print("Accuracy:",accuracy_score(y_test, y_pred))

lr_disp = plot_roc_curve(clf, X_test, y_test)
matrix = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(matrix).plot()
report = classification_report(y_test, y_pred)
print(report)