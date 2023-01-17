import pandas as pd 

import seaborn as sns           # for data visualization

import matplotlib.pyplot as plt # for data visualization

%matplotlib inline
df = pd.read_csv("../input/Breast_cancer_data.csv", delimiter=",")
df.head() #gives first 5 entries of a dataframe by default
df.columns
df.isnull().sum()
count = df.diagnosis.value_counts()

count
count.plot(kind='bar')

plt.title("Distribution of malignant(1) and benign(0) tumor")

plt.xlabel("Diagnosis")

plt.ylabel("count");
df.describe()

df.head()
y = df.diagnosis                          # M or B 

list = ['diagnosis']

x = df.drop(list,axis = 1 )

x.head()
data_dia = y

data = x

data_n_2 = (data - data.mean()) / (data.std())              # standardization



data = pd.concat([y,data_n_2.iloc[:,0:5]],axis=1)           #for the first 5 features. In case there might be more, several graphs should be made

data = pd.melt(data,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')

plt.figure(figsize=(10,10))

sns.violinplot(x="features", y="value", hue="diagnosis", data=data,split=True, inner="quart")

plt.xticks(rotation=90)
plt.figure(figsize=(10,10))

sns.boxplot(x="features", y="value", hue="diagnosis", data=data)

plt.xticks(rotation=90)
sns.set(style="whitegrid", palette="muted")

sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)



plt.xticks(rotation=90)
y_target = df['diagnosis']
df.columns.values
df['target'] = df['diagnosis'].map({0:'B',1:'M'}) # converting the data into categorical
g = sns.pairplot(df.drop('diagnosis', axis = 1), hue="target", palette='prism');
f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
sns.jointplot(x.loc[:,'mean_perimeter'], x.loc[:,'mean_radius'], kind="regg", color="#ce1414"); 

sns.jointplot(x.loc[:,'mean_perimeter'], x.loc[:,'mean_area'], kind="regg", color="#ce1414"); 
sns.scatterplot(x='mean_perimeter', y = 'mean_texture', data = df, hue = 'target', palette='prism')
features = ['mean_perimeter', 'mean_texture']
X_feature = df[features]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_feature, y_target, test_size=0.3, random_state = 42)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score



#random forest classifier with n_estimators=10 (default)

clf_rf = RandomForestClassifier(random_state=43)      

clr_rf = clf_rf.fit(X_train,y_train)



ac = accuracy_score(y_test,clf_rf.predict(X_test))

print('Accuracy is: ',ac)

cm = confusion_matrix(y_test,clf_rf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

# find best scored 1 feature

select_feature = SelectKBest(chi2, k=1).fit(X_train, y_train)

print('Score list:', select_feature.scores_)

print('Feature list:', X_train.columns)



X_train_2 = select_feature.transform(X_train)

X_test_2 = select_feature.transform(X_test)

#random forest classifier with n_estimators=10 (default)

clf_rf_2 = RandomForestClassifier()      

clr_rf_2 = clf_rf_2.fit(X_train_2,y_train)

ac_2 = accuracy_score(y_test,clf_rf_2.predict(X_test_2))

print('Accuracy is: ',ac_2)

cm_2 = confusion_matrix(y_test,clf_rf_2.predict(X_test_2))

sns.heatmap(cm_2,annot=True,fmt="d")
from sklearn.feature_selection import RFE

# Create the RFE object and rank each pixel

clf_rf_3 = RandomForestClassifier()      

rfe = RFE(estimator=clf_rf_3, n_features_to_select=1, step=1)

rfe = rfe.fit(X_train, y_train)

print('Chosen best feature by rfe:',X_train.columns[rfe.support_])
from sklearn.feature_selection import RFECV



# The "accuracy" scoring is proportional to the number of correct classifications

clf_rf_4 = RandomForestClassifier() 

rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5,scoring='accuracy')   #5-fold cross-validation

rfecv = rfecv.fit(X_train, y_train)



print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])



# Plot number of features VS. cross-validation scores

import matplotlib.pyplot as plt

plt.figure()

plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score of number of selected features")

plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)

plt.show()
import numpy as np # linear algebra



clf_rf_5 = RandomForestClassifier()      

clr_rf_5 = clf_rf_5.fit(X_train,y_train)

importances = clr_rf_5.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



# Print the feature ranking

print("Feature ranking:")



for f in range(X_train.shape[1]):

    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))



# Plot the feature importances of the forest



plt.figure(1, figsize=(14, 13))

plt.title("Feature importances")

plt.bar(range(X_train.shape[1]), importances[indices],

       color="g", yerr=std[indices], align="center")

plt.xticks(range(X_train.shape[1]), X_train.columns[indices],rotation=90)

plt.xlim([-1, X_train.shape[1]])

plt.show()
#normalization

X_train_N = (X_train-X_train.mean())/(X_train.max()-X_train.min())

X_test_N = (X_test-X_test.mean())/(X_test.max()-X_test.min())



from sklearn.decomposition import PCA

pca = PCA()

pca.fit(X_train_N)



plt.figure(1, figsize=(14, 13))

plt.clf()

plt.axes([.2, .2, .7, .7])

plt.plot(pca.explained_variance_ratio_, linewidth=2)

plt.axis('tight')

plt.xlabel('n_components')

plt.ylabel('explained_variance_ratio_')
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
model = LogisticRegression()

model.fit(X_train, y_train)

from mlxtend.plotting import plot_decision_regions

plot_decision_regions(X_train.values, y_train.values, clf=model, legend=2)

plt.title("Decision boundary for Logistic Regression (Train)")

plt.xlabel("mean_perimeter")

plt.ylabel("mean_texture");



y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy score using Logistic Regression:", acc*100, "This is better than using random forest (90%)!")

from sklearn.metrics import confusion_matrix

conf_mat = confusion_matrix(y_test, y_pred)

conf_mat
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("Accuracy score using KNN:", acc*100, "This is less good than using logistic regression (93%)")

confusion_matrix(y_test, y_pred)



plot_decision_regions(X_train.values, y_train.values, clf=clf, legend=2)

plt.title("Decision boundary using KNN (Train)")

plt.xlabel("mean_perimeter")

plt.ylabel("mean_texture");