import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.datasets import make_classification



from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler

from imblearn.over_sampling import SMOTE



from sklearn import (tree, metrics)

from sklearn.model_selection import (train_test_split, cross_validate) 







data = pd.read_csv("../input/creditcard.csv")

data["Time"] = data["Time"].apply(lambda x : x / 3600 % 24)

data.head(5)
fraud_class = np.array(data.groupby(['Class']).agg({'Class': 'count'}))

fraud_class
rows = data.Class.shape[0]
(fraud_class[0] / rows) 
(fraud_class[1] / rows)
x = data.drop('Class', axis=1)

y = data['Class']



maj_under10 = round(data[data['Class']!=1].shape[0]*.1)

maj_under100 = round(data[data['Class']!=1].shape[0])

all_frd_cnt = data[data['Class']==1].shape[0]



all_frd_cnt_thrice = all_frd_cnt * 3
all_frd_cnt
# Apply the random under-sampling

rus = RandomUnderSampler(ratio={0: maj_under10, 1: all_frd_cnt})

x_RUS, y_RUS = rus.fit_sample(x, y)

# Apply the random over-sampling

ros = RandomOverSampler(ratio={0: maj_under100, 1: all_frd_cnt_thrice})

x_ROS, y_ROS = ros.fit_sample(x, y)
x_np = np.array(x)

y_np = np.array(y)
# Original vs resampled subplots

plt.figure(figsize=(20, 8))

plt.scatter(x_np[y_np==0,0], x_np[y_np==0,1], marker='o', color='blue')

plt.scatter(x_np[y_np==1,0], x_np[y_np==1,1], marker='+', color='red')

plt.xlabel('x1')

plt.ylabel('x2')

plt.title('Original: 1=%s and 0=%s' %(y_np.tolist().count(1), y_np.tolist().

count(0)))
plt.figure(figsize=(20, 8))



plt.scatter(x_RUS[y_RUS==0,0], x_RUS[y_RUS==0,1], marker='o', color='blue')

plt.scatter(x_RUS[y_RUS==1,0], x_RUS[y_RUS==1,1], marker='+', color='red')

plt.xlabel('x1')

plt.ylabel('y2')

plt.title('Random Under-sampling: 1=%s and 0=%s' %(y_RUS.tolist().count(1),

y_RUS.tolist().count(0)))
plt.figure(figsize=(20, 8))

plt.scatter(x_ROS[y_ROS==0,0], x_ROS[y_ROS==0,1], marker='o', color='blue')

plt.scatter(x_ROS[y_ROS==1,0], x_ROS[y_ROS==1,1], marker='+', color='red')

plt.xlabel('x1')

plt.ylabel('x2')

plt.title('Random over-sampling: 1=%s and 0=%s' %(y_ROS.tolist().count(1),

y_ROS.tolist().count(0)))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2017)

x_RUS_train, x_RUS_test, y_RUS_train, y_RUS_test = train_test_split(x_RUS, y_RUS, test_size=0.3, random_state=2017)

x_ROS_train, x_ROS_test, y_ROS_train, y_ROS_test = train_test_split(x_ROS, y_ROS, test_size=0.3, random_state=2017)

# build a decision tree classifier

clf = tree.DecisionTreeClassifier(random_state=2017)

clf_orig= clf.fit(x_train, y_train)

clf_rus = clf.fit(x_RUS_train, y_RUS_train)

clf_ros = clf.fit(x_ROS_train, y_ROS_train)

# evaluate model performance

print ("Original - Train AUC : ",metrics.roc_auc_score(y_train, clf.predict(x_train)))

print ("Original - Test AUC : ",metrics.roc_auc_score(y_test, clf.predict(x_test)))

print ("RUS - Train AUC : ",metrics.roc_auc_score(y_RUS_train, clf.predict(x_RUS_train)))

print ("RUS - Test AUC : ",metrics.roc_auc_score(y_RUS_test, clf.predict(x_RUS_test)))

print ("ROS - Train AUC : ",metrics.roc_auc_score(y_ROS_train, clf.predict(x_ROS_train)))

print ("ROS - Test AUC : ",metrics.roc_auc_score(y_ROS_test, clf.predict(x_ROS_test)))
# generate evaluation metrics

print ("Original - Train AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_train, clf.predict(x_train)))

print ("classification report\n ", metrics.classification_report(y_train, clf.predict(x_train)))



print ("Original - Test AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_test, clf.predict(x_test)))

print ("classification report\n ", metrics.classification_report(y_test, clf.predict(x_test)))



print ("RUS - Train AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_RUS_train, clf.predict(x_RUS_train)))

print ("classification report\n ", metrics.classification_report(y_RUS_train, clf.predict(x_RUS_train)))



print ("RUS - Test AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_RUS_test, clf.predict(x_RUS_test)))

print ("classification report\n ", metrics.classification_report(y_RUS_test, clf.predict(x_RUS_test)))



print ("ROS - Train AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_ROS_train, clf.predict(x_ROS_train)))

print ("classification report\n ", metrics.classification_report(y_ROS_train, clf.predict(x_ROS_train)))



print ("ROS - Test AUC : ")

print ("classification report\n ", metrics.confusion_matrix(y_ROS_test, clf.predict(x_ROS_test)))

print ("classification report\n ", metrics.classification_report(y_ROS_test, clf.predict(x_ROS_test)))
final_sub = metrics.classification_report(y_RUS_train, clf.predict(x_RUS_train))

temp.to_csv("submission.csv", index = False)

temp.head()
final_sub = clf.predict(x_RUS_train)

fina_df = pd.DataFrame(final_sub)

fina_df.to_csv("submission.csv", index = False)

fina_df.head()
