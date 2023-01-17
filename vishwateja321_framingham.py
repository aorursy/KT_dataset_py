import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True,style='darkgrid')

file  = '/kaggle/input/framingham-heart-study-dataset/framingham.csv'
df = pd.read_csv(file)

df.head()
df.describe(include='all').transpose()
df.isnull().sum()
df.dtypes
from sklearn.impute import KNNImputer
knn = KNNImputer(n_neighbors=1)
missing_cols = df[['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose']]
missing = np.array(missing_cols)
imputed_cols = knn.fit_transform(missing)
imputed_cols = pd.DataFrame(imputed_cols,columns=['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose'])
imputed_cols.head()
df = df.drop(['education','cigsPerDay','BPMeds','totChol','BMI','heartRate','glucose'],axis=1)
df.head()
df = pd.concat([df,imputed_cols],axis=1)
df.head()
df.columns
df.isnull().sum()
df['TenYearCHD'].value_counts()
sns.distplot(df['age'])
sns.distplot(df['BMI'])
sns.distplot(df['heartRate'])
sns.distplot(df['glucose'])
sns.distplot(df['totChol'])
sns.countplot(df['education'],hue=df['TenYearCHD'])
df.education.value_counts()
sns.countplot(df['currentSmoker'],hue=df['TenYearCHD'])
sns.boxplot(df['TenYearCHD'],df['totChol'])
sns.pairplot(df)
corr = df.corr()
plt.figure(figsize=(15,10))
sns.heatmap(corr,annot =True )
X = df.drop(['TenYearCHD'],axis=1)
y = df['TenYearCHD']
from imblearn.over_sampling import RandomOverSampler
os = RandomOverSampler(0.5)
x_os,y_os = os.fit_sample(X,y)
x_os.shape
y_os.shape
type(y_os)
x_cols = X.columns
x_os = pd.DataFrame(x_os,columns=x_cols)
y_os = pd.Series(y_os,name='TenYearCHD')
y_os.shape
x_os.shape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train,x_test,y_train,y_test = train_test_split(x_os,y_os,test_size=0.3,random_state=0)
x_train.shape
x_test.shape
y_test.value_counts()
scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(class_weight='balanced',random_state=0)
model1 = logit.fit(xs_train,y_train)
y_pred1 = model1.predict(xs_test)
xs_test.shape
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,confusion_matrix
accuracy_score(y_test,y_pred1)
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)
y_proba1 = model1.predict_proba(xs_test)[:,1]
confusion_matrix(y_test,y_pred1)
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba1)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1],ls = '--')
roc_auc_score(y_test,y_proba1)
from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
param = {'n_neighbors':[3,5,7,9,11],'weights':['uniform','distance']}
knc_clf = GridSearchCV(knc,param_grid=param,scoring='recall',cv=5,n_jobs=-1,verbose=3).fit(xs_train,y_train)
knc_clf.best_params_
model2 = KNeighborsClassifier(n_neighbors=7,weights='distance',).fit(xs_train,y_train)
y_pred2 = model2.predict(xs_test)
accuracy_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
f1_score(y_test,y_pred2)
y_proba2 = model2.predict_proba(xs_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba2)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1],ls='--')
roc_auc_score(y_test,y_proba2)
confusion_matrix(y_test,y_pred2)
from sklearn.svm import LinearSVC
svc = LinearSVC()
model3 = svc.fit(xs_train,y_train)
y_pred3 = model3.predict(xs_test)
accuracy_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
f1_score(y_test,y_pred3)
from sklearn.svm import SVC
svm = SVC(C=750,probability=True,class_weight='balanced')
model_3 = svm.fit(xs_train,y_train)
y3_pred = model_3.predict(xs_test)
accuracy_score(y_test,y3_pred)
recall_score(y_test,y3_pred)
y_proba3 = model_3.predict_proba(xs_test)[:,1]
confusion_matrix(y_test,y3_pred)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion='gini',class_weight='balanced')
model4 = dtree.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
accuracy_score(y_test,y_pred4)
recall_score(y_test,y_pred4)
f1_score(y_test,y_pred4)
confusion_matrix(y_test,y_pred4)
y_proba4 = model4.predict_proba(x_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba4)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1])
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.show()
roc_auc_score(y_test,y_proba4)
from xgboost import XGBClassifier
model5 = XGBClassifier(learning_rate=0.3,colsample_bynode=0.8,subsample=0.8).fit(x_train,y_train)
y_pred5 = model5.predict(x_test)
accuracy_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
f1_score(y_test,y_pred5)
y_proba5 = model5.predict_proba(x_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba5)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1])
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.show()
roc_auc_score(y_test,y_proba5)
from lightgbm import LGBMClassifier
lgb_clf = LGBMClassifier(class_weight='balanced',n_jobs=-1,n_estimators=300,subsample=0.6,colsample_bytree=0.5)
model6 = lgb_clf.fit(x_train,y_train)
y_pred6 = model6.predict(x_test)
accuracy_score(y_test,y_pred6)
recall_score(y_test,y_pred6)
f1_score(y_test,y_pred6)
y_proba6 = model6.predict_proba(x_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba6)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1])
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.show()
roc_auc_score(y_test,y_proba6)
confusion_matrix(y_test,y_pred6)
yp6 = model6.predict_proba(x_test)
from sklearn.preprocessing import binarize
y6_pred = binarize(yp6,0.3)[:,1]
recall_score(y_test,y6_pred)
accuracy_score(y_test,y6_pred)
confusion_matrix(y_test,y6_pred)
from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(class_weight='balanced',n_estimators=250,max_features='auto')
model8 = rfcl.fit(x_train,y_train)
y_pred8 = model8.predict(x_test)
accuracy_score(y_test,y_pred8)
recall_score(y_test,y_pred8)
f1_score(y_test,y_pred8)
y_proba8 = model8.predict_proba(x_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba8)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1])
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.show()
roc_auc_score(y_test,y_proba8)
confusion_matrix(y_test,y_pred8)
from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.3,subsample=0.5,max_features=1.0)
model9 = gbcl.fit(x_train,y_train)
y_pred9 = model9.predict(x_test)
accuracy_score(y_test,y_pred9)
recall_score(y_test,y_pred9)
f1_score(y_test,y_pred9)
confusion_matrix(y_test,y_pred9)
y_proba9 = model9.predict_proba(x_test)[:,1]
false_positive_rate,true_positive_rate,threshold = roc_curve(y_test,y_proba9)
plt.plot(false_positive_rate,true_positive_rate)
plt.plot([0,1])
plt.xlabel('false_positive_rate')
plt.ylabel('true_positive_rate')
plt.show()
roc_auc_score(y_test,y_proba9)
