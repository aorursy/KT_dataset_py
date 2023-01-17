import pandas as pd

import numpy as np
#Scaling data

from sklearn.preprocessing import StandardScaler as ss

#Importing PCA class

from sklearn.decomposition import PCA

#For data splitting

from sklearn.model_selection import train_test_split

#For modeling

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost.sklearn import XGBClassifier

from sklearn.neighbors import KNeighborsClassifier

#For performance measures

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import auc, roc_curve

from sklearn.metrics import precision_recall_fscore_support

#For plotting

import matplotlib.pyplot as plt
df = pd.read_csv("../input/data.csv")
df.shape
df.head()
#Checking if any column has any missing variable

df.isnull().sum()
# Dropping Columns that are not needed          

df=df.drop(columns = "id")

df=df.drop(columns = "Unnamed: 32")
df.shape
df.columns.values
#Segregating dataset into predictors (X) 

X = df.iloc[:, 1:31]
X
#Segregating dataset into Target (Y) 

y = df.loc[:,['diagnosis']].values
y
#Mapping values in ' y ' (target) from 'M' and 'B' to 1 and 0

y[y=='M'] = 1

y[y=='B'] = 0
y
y = y.astype('int64') 
#Scale all numerical features in X  using sklearn's StandardScaler class

scale = ss()

X = scale.fit_transform(X)
X.shape
pca = PCA()

out = pca.fit_transform(X)
out.shape
pca.explained_variance_ratio_

pca.explained_variance_ratio_.cumsum()
#Retaining as many principal components (PCs) as explain 95% variance

X = out[:, :10]
y = y.ravel()
#Split Data

X_train, X_test, y_train, y_test = train_test_split(X,

                                                    y,

                                                    test_size = 0.2,

                                                    shuffle = True

                                                    )
#Creating default classifiers

dt = DecisionTreeClassifier()

rf = RandomForestClassifier(n_estimators=100)

et = ExtraTreesClassifier(n_estimators=100)

gbm = GradientBoostingClassifier()

xg = XGBClassifier(learning_rate=0.5,

                   reg_alpha= 5,

                   reg_lambda= 0.1)

knn = KNeighborsClassifier()
#Training data

dt1 = dt.fit(X_train,y_train)

rf1 = rf.fit(X_train,y_train)

et1 = et.fit(X_train,y_train)

gbm1 = gbm.fit(X_train,y_train)

xg1 = xg.fit(X_train,y_train)

knn1 = knn.fit(X_train,y_train)
#Making predictions

y_pred_dt = dt1.predict(X_test)

y_pred_rf = rf1.predict(X_test)

y_pred_et = et1.predict(X_test)

y_pred_gbm = gbm1.predict(X_test)

y_pred_xg = xg1.predict(X_test)

y_pred_knn = knn1.predict(X_test)
#Getting probability values

y_pred_dt_prob = dt1.predict_proba(X_test)

y_pred_rf_prob = rf1.predict_proba(X_test)

y_pred_et_prob = et1.predict_proba(X_test)

y_pred_gbm_prob = gbm1.predict_proba(X_test)

y_pred_xg_prob = xg1.predict_proba(X_test)

y_pred_knn_prob = knn1.predict_proba(X_test)
#Calculating accuracy

print("Accuracy score of Decision Tree Classifier is :" ,accuracy_score(y_test,y_pred_dt))

print("Accuracy score of Random Forest Classifier is :" ,accuracy_score(y_test,y_pred_rf))

print("Accuracy score of Extra Trees Classifier is :" ,accuracy_score(y_test,y_pred_et))

print("Accuracy score of Gradient Boost Classifier is :" ,accuracy_score(y_test,y_pred_gbm))

print("Accuracy score of XG Boost Classifier is :" ,accuracy_score(y_test,y_pred_xg))

print("Accuracy score of KNeighbors Classifier is :" ,accuracy_score(y_test,y_pred_knn))
# Calculating Precision/Recall/F-score

p_dt,r_dt,f_dt,_ = precision_recall_fscore_support(y_test,y_pred_dt)

print("The precision, recall and fscore of Decision Tree Classifier are :", p_dt,r_dt,f_dt)

p_rf,r_rf,f_rf,_ = precision_recall_fscore_support(y_test,y_pred_rf)

print("The precision, recall and fscore of Random Forest Classifier are :", p_rf,r_rf,f_rf)

p_et,r_et,f_et,_ = precision_recall_fscore_support(y_test,y_pred_et)

print("The precision, recall and fscore of Extra Trees Classifier are :", p_et,r_et,f_et)

p_gbm,r_gbm,f_gbm,_ = precision_recall_fscore_support(y_test,y_pred_gbm)

print("The precision, recall and fscore of Gradient Boost Classifier are :", p_gbm,r_gbm,f_gbm)

p_xg,r_xg,f_xg,_ = precision_recall_fscore_support(y_test,y_pred_xg)

print("The precision, recall and fscore of XG Boost Classifier are :", p_xg,r_xg,f_xg)

p_knn,r_knn,f_knn,_ = precision_recall_fscore_support(y_test,y_pred_knn)

print("The precision, recall and fscore of KNeighbors Classifier are :", p_knn,r_knn,f_knn)
#Drawing Confusion matrix

print("Confusion matrix of Decision Tree Classifier is :" ,confusion_matrix(y_test,y_pred_dt))

print("Confusion matrix of Random Forest Classifier is :" ,confusion_matrix(y_test,y_pred_rf))

print("Confusion matrix of Extra Trees Classifier is :" ,confusion_matrix(y_test,y_pred_et))

print("Confusion matrix of Gradient Boost Classifier is :" ,confusion_matrix(y_test,y_pred_gbm))

print("Confusion matrix of XG Boost Classifier is :" ,confusion_matrix(y_test,y_pred_xg))

print("Confusion matrix of KNeighbors Classifier is :" ,confusion_matrix(y_test,y_pred_knn))
#FPR and TPR Values

fpr_dt, tpr_dt, thresholds = roc_curve(y_test,

                                 y_pred_dt_prob[: , 1],

                                 pos_label= 1

                                 )



fpr_rf, tpr_rf, thresholds = roc_curve(y_test,

                                 y_pred_rf_prob[: , 1],

                                 pos_label= 1

                                 )



fpr_et, tpr_et, thresholds = roc_curve(y_test,

                                 y_pred_et_prob[: , 1],

                                 pos_label= 1

                                 )

fpr_gbm, tpr_gbm, thresholds = roc_curve(y_test,

                                 y_pred_gbm_prob[: , 1],

                                 pos_label= 1

                                 )



fpr_xg, tpr_xg, thresholds = roc_curve(y_test,

                                 y_pred_xg_prob[: , 1],

                                 pos_label= 1

                                 )



fpr_knn, tpr_knn,thresholds = roc_curve(y_test,

                                 y_pred_knn_prob[: , 1],

                                 pos_label= 1

                                 )
#Calculating AUC values

print("The AUC Value of Decision Tree Classifier is :", auc(fpr_dt,tpr_dt))

print("The AUC Value of Random Forest Classifier is :",auc(fpr_rf,tpr_rf))

print("The AUC Value of Extra Trees Classifier is :",auc(fpr_et,tpr_et))

print("The AUC Value of Gradient Boost Classifier is :",auc(fpr_gbm,tpr_gbm))

print("The AUC Value of XG Boost Classifier is :",auc(fpr_xg,tpr_xg))

print("The AUC Value of KNeighbors Classifier is :",auc(fpr_knn,tpr_knn))
# Plotting Graph

fig = plt.figure(figsize=(12,10))          # Create window frame

ax = fig.add_subplot(111)   # Create axes

# connecting diagonals

ax.plot([0, 1], [0, 1], ls="--")   # Dashed diagonal line

# Creating Labels for Graph

ax.set_xlabel('False Positive Rate')  # Final plot decorations

ax.set_ylabel('True Positive Rate')

ax.set_title('ROC curve for models')

# Setting graph limits

ax.set_xlim([0.0, 1.0])

ax.set_ylim([0.0, 1.0])



# Plotting each graph

ax.plot(fpr_dt, tpr_dt, label = "Decision Tree")

ax.plot(fpr_rf, tpr_rf, label = "Random Forest")

ax.plot(fpr_et, tpr_et, label = "Extra Trees")

ax.plot(fpr_gbm, tpr_gbm, label = "Gradient Boost")

ax.plot(fpr_xg, tpr_xg, label = "XG Boost")

ax.plot(fpr_knn, tpr_knn, label = "KNeighbors")



# Setting legend and show plot

ax.legend(loc="lower right")

plt.show()
print(" Gradient Boost Classfier has the best performance")
print(" Observation : As the FPR increase is faster in Decision Tree classifier, it's AUC is less compared to Gradient Boost Classifier")
