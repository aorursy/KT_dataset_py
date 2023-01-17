import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
breast_cancer_data = pd.read_csv('../input/data.csv')
breast_cancer_data.shape
breast_cancer_data.head()
breast_cancer_data.describe()
breast_cancer_data.info()
breast_cancer_data = breast_cancer_data.drop(['id','Unnamed: 32'], axis=1)
breast_cancer_data.describe()
# Correlation between all the features

breast_cancer_data.corr()
# Remove the highly co-related features



breast_cancer_data = breast_cancer_data.drop(['radius_mean','perimeter_mean','concave points_mean','radius_se','radius_worst','perimeter_worst','perimeter_se','concave points_worst','compactness_worst','texture_worst','area_worst','concavity_worst'], axis=1)
# Now we have the feauters which are least related to each-other

breast_cancer_data.corr()
breast_cancer_data.shape

n_features = breast_cancer_data.shape[1]
breast_cancer_data.head()

target = breast_cancer_data['diagnosis']

breast_cancer_data = breast_cancer_data.drop(['diagnosis'], axis=1)
# Split train and test data

x_train, x_test, y_train, y_test = train_test_split(breast_cancer_data, target)
# Find the depth which gives high accuracy

for depth in range(1,20):

    clf = RandomForestClassifier(max_depth = depth)

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print('Accuracy is: {} at depth {}'.format(accuracy_score(y_test, y_pred), depth))

    #print('Score is: ',clf.score(x_test,y_test))
# Classification

clf = RandomForestClassifier(max_depth = 5)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print('Accuracy is: {}'.format(accuracy_score(y_test, y_pred)))
# Confusion Matrix

confusion_matrix = confusion_matrix(y_test,y_pred)

confusion_matrix
clf_dt = tree.DecisionTreeClassifier(max_depth = 5)

clf_dt.fit(x_train,y_train)

y_pred_dt = clf_dt.predict(x_test)

print('Accuracy is: {}'.format(accuracy_score(y_test, y_pred_dt)))