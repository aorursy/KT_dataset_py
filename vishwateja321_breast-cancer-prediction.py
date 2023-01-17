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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid',color_codes=True)
file = "/kaggle/input/breast-cancer-prediction-dataset/Breast_cancer_data.csv"
df = pd.read_csv(file)
df.head()
df.shape
df.isna().sum()
df['diagnosis'].value_counts()
df.describe()
sns.boxplot(df['mean_area'])
from sklearn.covariance import EllipticEnvelope
outlier_detector = EllipticEnvelope(contamination=0.05)
outlier_detector.fit(df)
in_out = outlier_detector.predict(df)
out = np.where(in_out == -1)
out
df.loc[out]
for feature in df.columns:
    plt.figure(figsize=(15,5))
    sns.distplot(df[feature])
    plt.xlim(df[feature].min(),df[feature].max())
    plt.title(f"Distribution shape of {feature.capitalize()}\n",fontsize=15)
    plt.show()
sns.countplot(df['diagnosis'])
sns.boxplot(df['diagnosis'],df['mean_radius'])
sns.boxplot(df['diagnosis'],df['mean_area'])
sns.boxplot(df['diagnosis'],df['mean_perimeter'])
sns.boxplot(df['diagnosis'],df['mean_smoothness'])
sns.boxplot(df['diagnosis'],df['mean_texture'])
df.groupby(['diagnosis'])['mean_radius'].mean().plot(kind='bar')
df.groupby(['diagnosis'])['mean_texture'].median().plot(kind='bar')
X = df.drop(['diagnosis'],axis=1)
y = df['diagnosis']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
x_train.shape
y_test.value_counts()
y_train.value_counts()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=0,class_weight='balanced')
model1 = lr.fit(xs_train,y_train)
y_pred1 = model1.predict(xs_test)
from sklearn.metrics import accuracy_score,recall_score,f1_score,precision_score,roc_auc_score,roc_curve,confusion_matrix
accuracy_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
confusion_matrix(y_test,y_pred1)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,weights = 'distance')
model2 = knn.fit(xs_train,y_train)
y_pred2 = model2.predict(xs_test)
accuracy_score(y_test,y_pred2)
f1_score(y_test,y_pred2)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='entropy',random_state=0)
model3 = dtree.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)
accuracy_score(y_test,y_pred3)
f1_score(y_test,y_pred3)
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',class_weight='balanced')
model4 = rf.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
accuracy_score(y_test,y_pred4)
f1_score(y_test,y_pred4)
