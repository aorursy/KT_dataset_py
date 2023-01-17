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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True,style='darkgrid')
file = "/kaggle/input/vehicle1/vehicle-1.csv"
df = pd.read_csv(file)
df.head()
df.shape
df['class'].unique()
df.info()
df.isna().sum()
df.describe(include='all').transpose()
for feature in df.select_dtypes("number").columns:
    sns.distplot(df[feature])
    plt.title(f"Distribution of {feature.capitalize()}\n")
    plt.tight_layout()
    plt.show()
for feature in df.select_dtypes("number").columns:
    sns.boxplot(df['class'],df[feature])
    plt.tight_layout()
    plt.show()
df1 = df.replace(['van','car','bus'],[1,0,2])
df1.head()
sns.pairplot(df1,diag_kind='kde')
df.isna().sum().sum()
null_data = df[df.isnull().any(axis=1)]
null_data
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5,weights='distance')
df1 = imputer.fit_transform(df1)
cols = df.columns
df1 = pd.DataFrame(df1,columns=cols)
df1.head()
df1.isna().sum()
df1.skew()
df[df['max.length_aspect_ratio'] >= 20]
df[df['pr.axis_aspect_ratio'] >= 82]
df1['pr.axis_aspect_ratio'] = np.log(df1['pr.axis_aspect_ratio'])
df1['pr.axis_aspect_ratio'].skew()
df1['max.length_aspect_ratio'] = np.log(df1['max.length_aspect_ratio'])
df1['max.length_aspect_ratio'].skew()
df1['scaled_radius_of_gyration.1'] = np.log(df1['scaled_radius_of_gyration.1'])
df1['scaled_radius_of_gyration.1'].skew()
corr = df1.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)
X = df1.drop(['class'],axis=1)
y = df1['class']
X.head()
from sklearn.ensemble import ExtraTreesClassifier
ensemble = ExtraTreesClassifier(random_state=0)
fit = ensemble.fit(X,y)
imp = fit.feature_importances_
score = pd.DataFrame(X.columns,columns=['feature'])
score['scores'] = imp
score.sort_values(by='scores',ascending=False)
X = X.drop(['scaled_radius_of_gyration','skewness_about','skewness_about.1'],axis=1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
x_train.shape
x_test.shape
y_test.value_counts()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)
Xscaled = scaler.fit_transform(X)
from sklearn.svm import LinearSVC
lsvc = LinearSVC(C=1,max_iter=5000)
model1 = lsvc.fit(xs_train,y_train)
y_pred1 = model1.predict(xs_test)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
acc = accuracy_score(y_test,y_pred1)
acc
f1_score(y_test,y_pred1,average='weighted')
confusion_matrix(y_test,y_pred1)
from sklearn.svm import SVC
svc = SVC(kernel='rbf',random_state=0)
model2 = svc.fit(xs_train,y_train)
y_pred2 = model2.predict(xs_test)
rbf_acc = accuracy_score(y_test,y_pred2)
rbf_acc
f1_score(y_test,y_pred2,average='macro')
confusion_matrix(y_test,y_pred2)
from sklearn.model_selection import cross_val_score,KFold
kf = KFold(n_splits=10,shuffle=True,random_state=1)
cv = cross_val_score(lsvc,Xscaled,y,cv=kf,scoring='accuracy',n_jobs=-1)
cv
cv_acc = cv.mean()
cv_acc
cv.std()
cv_ = cross_val_score(svc,Xscaled,y,cv=kf,scoring='accuracy',n_jobs=-1)
cv_
rbf_cv_acc = cv_.mean()
rbf_cv_acc
cv_.std()
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(Xscaled)
pca.explained_variance_
pca.explained_variance_ratio_
plt.figure(figsize=(10,5))
plt.bar(list(range(1,16)),pca.explained_variance_ratio_)
plt.figure(figsize=(10,5))
plt.step(list(range(1,16)),np.cumsum(pca.explained_variance_ratio_))
reduced_pca = PCA(n_components=0.95)
reduced_pca.fit(Xscaled)
reduced_pca.components_
reduced_pca.explained_variance_
reduced_pca.explained_variance_ratio_
Xpca = reduced_pca.transform(Xscaled)
xp_train,xp_test,y_train,y_test = train_test_split(Xpca,y,test_size=0.3,random_state=42)
model3 = LinearSVC(max_iter=5000).fit(xp_train,y_train)
y_pred3 = model3.predict(xp_test)
pca_acc = accuracy_score(y_test,y_pred3)
pca_acc
f1_score(y_test,y_pred3,average='weighted')
confusion_matrix(y_test,y_pred3)
model4 = SVC().fit(xp_train,y_train)
y_pred4 = model4.predict(xp_test)
pca_rbf_acc = accuracy_score(y_test,y_pred4)
pca_rbf_acc
confusion_matrix(y_test,y_pred4)
p_cv = cross_val_score(lsvc,Xpca,y,cv=kf,scoring='accuracy',n_jobs=-1)
p_cv
pca_cv_acc = p_cv.mean()
pca_cv_acc
p_cv.std()
p_cv_ = cross_val_score(svc,Xpca,y,cv=kf,scoring='accuracy',n_jobs=-1)
p_cv_
pca_rbf_cv_acc = p_cv_.mean()
pca_rbf_cv_acc
p_cv_.std()
print("Accracy scores on raw Data")
print("")
print("Accuracy of linear support vector classifier:",round(acc*100,2))
print("Accuracy of rbf kernel support vector calssifier:",round(rbf_acc*100,2))
print("Mean Accuraccy of linear support Vector Classifier on Cross Validation:",round(cv_acc*100,2))
print("Mean Accuracy of rbf kernel support vector Classifier on Cross Validation:",round(rbf_cv_acc*100,2))
print("Accracy scores on pca reduced Data")
print("")
print("Accuracy of linear support vector classifier:",round(pca_acc*100,2))
print("Accuracy of rbf kernel support vector calssifier:",round(pca_rbf_acc*100,2))
print("Mean Accuraccy of linear support Vector Classifier on Cross Validation:",round(pca_cv_acc*100,2))
print("Mean Accuracy of rbf kernel support vector Classifier on Cross Validation:",round(pca_rbf_cv_acc*100,2))
