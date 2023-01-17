#Importing Data manipulation and plotting modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os
#Importing libraries for modeling

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import ExtraTreesClassifier 

from sklearn.neighbors import KNeighborsClassifier
#Importing libraries for Data pre-processing

from sklearn.preprocessing import StandardScaler
#Importing model for Dimentionality Reduction

from sklearn.decomposition import PCA
#Importing libraries for performance measures

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import accuracy_score

from sklearn.metrics import auc, roc_curve
#Importing libraries For data splitting

from sklearn.model_selection import train_test_split
os.chdir("../input")
data = pd.read_csv("data.csv")
pd.options.display.max_columns = 200
data.head()
data.tail()
data.shape
data.dtypes
data.describe()
sns.jointplot(x='radius_mean',y='perimeter_mean',data=data)
data.isnull().any()
data = data.drop(['id','Unnamed: 32'],axis=1)
data.shape
sns.countplot(x='diagnosis',data=data)
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(1,4,1)

ax1.set_xticklabels(labels = 'Radius mean', rotation=90)

#plt.hist(x=data.radius_mean,bins=10)

sns.boxenplot(x='diagnosis',y='radius_mean',data=data)

ax1 = fig.add_subplot(1,4,2)

sns.boxenplot(x='diagnosis',y='texture_mean',data=data)

ax1 = fig.add_subplot(1,4,3)

sns.boxenplot(x='diagnosis',y='perimeter_mean',data=data)

ax1 = fig.add_subplot(1,4,4)

sns.boxenplot(x='diagnosis',y='area_mean',data=data)





fig2 = plt.figure(figsize=(12,12))

ax2 = fig2.add_subplot(1,4,1)

sns.boxenplot(x='diagnosis',y='smoothness_mean',data=data)

ax2 = fig2.add_subplot(1,4,2)

sns.boxenplot(x='diagnosis',y='compactness_mean',data=data)

ax2 = fig2.add_subplot(1,4,3)

sns.boxenplot(x='diagnosis',y='concavity_mean',data=data)

ax2 = fig2.add_subplot(1,4,4)

sns.boxenplot(x='diagnosis',y='concave points_mean',data=data)



fig3 = plt.figure(figsize=(12,12))

ax3 = fig3.add_subplot(1,4,1)

sns.boxenplot(x='diagnosis',y='symmetry_mean',data=data)

ax3 = fig3.add_subplot(1,4,2)

sns.boxenplot(x='diagnosis',y='fractal_dimension_mean',data=data)
X=data.iloc[:,1:32]
X.head()
X.shape
y=data.iloc[:,0]
y.head()
data.diagnosis.unique()
y = y.map({'M':1, 'B' : 0})
y.dtype
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA()
principleComponents = pca.fit_transform(X)
principleComponents.shape
pca.explained_variance_ratio_
X = pca.explained_variance_ratio_.cumsum()
X
sns.distplot(X,bins=5)
X = principleComponents[:,0:11]
X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size=0.20,

                                                    shuffle = True

                                                    )
X_train.shape
X_test.shape
dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=100)

xg = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1

                   )

gbm = GradientBoostingClassifier()

etc = ExtraTreesClassifier(n_estimators=100)

knn = KNeighborsClassifier(n_neighbors=10)
dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)

etc1 = etc.fit(X_train,y_train)

knn1 = knn.fit(X_train,y_train)
y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_xg= xg1.predict(X_test)

y_pred_gbm= gbm1.predict(X_test)

y_pred_etc= etc1.predict(X_test)

y_pred_knn = knn1.predict(X_test)
y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_gbm_prob= gbm1.predict_proba(X_test)

y_pred_etc_prob= etc1.predict_proba(X_test)

y_pred_knn_prob= knn1.predict_proba(X_test)
accuracy_score(y_test,y_pred_dt)
accuracy_score(y_test,y_pred_rf)
accuracy_score(y_test,y_pred_xg)
accuracy_score(y_test,y_pred_gbm)
accuracy_score(y_test,y_pred_etc)
accuracy_score(y_test,y_pred_knn)
confusion_matrix(y_test,y_pred_dt)
sns.heatmap(confusion_matrix(y_test,y_pred_dt),annot=True)
confusion_matrix(y_test,y_pred_rf)
sns.heatmap(confusion_matrix(y_test,y_pred_rf),annot=True)
confusion_matrix(y_test,y_pred_xg)
sns.heatmap(confusion_matrix(y_test,y_pred_xg),annot=True)
confusion_matrix(y_test,y_pred_gbm)
sns.heatmap(confusion_matrix(y_test,y_pred_gbm),annot=True)
confusion_matrix(y_test,y_pred_etc)
sns.heatmap(confusion_matrix(y_test,y_pred_etc),annot=True)
confusion_matrix(y_test,y_pred_knn)
sns.heatmap(confusion_matrix(y_test,y_pred_knn),annot=True)
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_dt).flatten()
tn,fp,fn,tp
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_xg).flatten()
tn,fp,fn,tp
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_rf).flatten()
tn,fp,fn,tp
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_gbm).flatten()
tn,fp,fn,tp
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_etc).flatten()
tn,fp,fn,tp
tn,fp,fn,tp= confusion_matrix(y_test,y_pred_knn).flatten()
tn,fp,fn,tp
fpr_dt, tpr_dt, thresholds = roc_curve(y_test,

                                 y_pred_dt_prob[: , 1],

                                 pos_label= 1

                                 )

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for Decision Tree')

ax.plot(fpr_dt, tpr_dt, label = "dt")
fpr_xg, tpr_xg, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob[: , 1],

                                 pos_label= 1

                                 )

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for XGBoost')

ax.plot(fpr_xg, tpr_xg, label = "xg")
fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob[: , 1],

                                 pos_label= 1

                                 )



fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for Random Forest')

ax.plot(fpr_rf, tpr_rf, label = "rf")
fpr_gbm, tpr_gbm,thresholds = roc_curve(y_test,

                                 y_pred_gbm_prob[: , 1],

                                 pos_label= 1

                                 )

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for Gradient Boosting')

ax.plot(fpr_gbm, tpr_gbm, label = "gbm")
fpr_etc, tpr_etc,thresholds = roc_curve(y_test,

                                 y_pred_etc_prob[: , 1],

                                 pos_label= 1

                                 )

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for Extra Trees Classifier')

ax.plot(fpr_etc, tpr_etc, label = "etc")
fpr_knn, tpr_knn,thresholds = roc_curve(y_test,

                                 y_pred_knn_prob[: , 1],

                                 pos_label= 1

                                 )

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for K Neighbors Classifier')

ax.plot(fpr_knn, tpr_knn, label = "knn")
#AUC of decision tree

auc(fpr_dt,tpr_dt)
#AUC of Random forest

auc(fpr_rf,tpr_rf)
#AUC of Gradient boosting

auc(fpr_gbm,tpr_gbm)
#AUC of XG Boost

auc(fpr_xg,tpr_xg)
#AUC of Extra Tree Classifier

auc(fpr_etc, tpr_etc)
#AUC of KNeighbor Classifier

auc(fpr_knn, tpr_knn)
p_dt,r_dt,f_dt,_ = precision_recall_fscore_support(y_test,y_pred_dt)
p_dt,r_dt,f_dt,_
p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)
p_rf,r_rf,f_rf,_
p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)
p_gbm,r_gbm,f_gbm,_
p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)
p_xg,r_xg,f_xg,_
p_etc,r_etc,f_etc,_ = precision_recall_fscore_support(y_test,y_pred_etc)
p_etc,r_etc,f_etc,_
p_knn,r_knn,f_knn,_ = precision_recall_fscore_support(y_test,y_pred_knn)
p_knn,r_knn,f_knn,_
models = [(dt, "decisiontree"), (rf, "randomForest"), (gbm, "gradientboost"),(xg,"xgboost"),(etc,"extratreesclassifier"),(knn,"kneighborclassifier")]

#Plot the ROC curve

fig = plt.figure(figsize=(24,20))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

# Also connect diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

ax.set_xlabel('False Positive Rate')

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])

AUC = []

for clf,name in models:

    clf.fit(X_train,y_train)

    y_pred_prob = clf.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(y_test,

                                     y_pred_prob[: , 1],

                                     pos_label= 1

                                     )

    AUC.append((auc(fpr,tpr)))

    ax.plot(fpr, tpr, label = name)







ax.legend(loc="lower right")

plt.show()

AUC