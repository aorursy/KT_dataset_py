import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

sns.set(style='white', palette='deep')

width=0.35

%matplotlib inline





#Function

def autolabel(rects,ax, df): #autolabel

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{} ({:.2f}%)'.format(height, height*100/df.shape[0]),

                    xy = (rect.get_x() + rect.get_width()/2, height),

                    xytext= (0,3),

                    textcoords="offset points",

                    ha='center', va='bottom')
#Importing dataset

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
#Analising dataset

cancer.keys()

for i in np.arange(len(cancer.keys())):

    print(cancer[list(cancer.keys())[i]])
#Creating Dataframe

data=cancer['data']

data = np.c_[data,cancer['target']]

columns= np.append(cancer['feature_names'], ['target'])

df = pd.DataFrame(data,columns=columns)

df.head()
#Data analysis

statistical = df.describe()

malignant= df[df['target']==1]

benign= df[df['target']==0]

print('Percentage of Malignant Tumor: {:.2f}%'.format((len(malignant)/len(df)) * 100)) 

print('Percentage of Benign Tumor: {:.2f}%'.format((len(benign)/len(df)) * 100))
statistical
#Looking for null values

null_values = (df.isnull().sum()/len(df))*100

null_values = pd.DataFrame(null_values,columns=['% Null Values'])

null_values
#Visualizing target

labels= [cancer['target_names'][0],cancer['target_names'][1]]

ind=np.arange(len(labels))

fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(1,1,1)

rects1 = ax.bar(labels[0],len(malignant), width=width, edgecolor='k')

rects2 = ax.bar(labels[1],len(benign), width=width, edgecolor='k' )

ax.set_xticks(ind)

ax.set_xlabel('Type of Cancer')

ax.set_ylabel('Quantity')

ax.grid(b=True,which='major', linestyle='--')

autolabel(rects1,ax,df)

autolabel(rects2,ax,df)
#Visualizing correlation

sns.pairplot(df, hue = 'target', vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
## Correlation with independent Variable (Note: Models like RF are not linear like these)

df2 = df.drop(['target'], axis=1)

df2.corrwith(df.target).plot.bar(

        figsize = (10, 10), title = "Correlation with Cancer", fontsize = 15,

        rot = 45, grid = True)
#Splitting X and y dataset

X = df.drop('target', axis=1)

y = df['target']

#Splitting the Dataset into the training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X.columns.values)

X_test = pd.DataFrame(sc_x.transform(X_test), columns=X.columns.values)
#### Model Building ####

### Comparing Models

## Logistic Regression

from sklearn.linear_model import LogisticRegression

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')

lr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = lr_classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
## K-Nearest Neighbors (K-NN)

#Choosing the K value

error_rate= []

for i in range(1,40):

    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')

print(np.mean(error_rate))
from sklearn.neighbors import KNeighborsClassifier

kn_classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)

kn_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = kn_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (Linear)

from sklearn.svm import SVC

svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)

svm_linear_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svm_linear_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## SVM (rbf)

from sklearn.svm import SVC

svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)

svm_rbf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = svm_rbf_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Naive Bayes

from sklearn.naive_bayes import GaussianNB

gb_classifier = GaussianNB()

gb_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gb_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)


## Decision Tree

from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

dt_classifier.fit(X_train, y_train)



#Predicting the best set result

y_pred = dt_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Random Forest

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 300,

                                    criterion = 'entropy')

rf_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = rf_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest Entropy (n=300)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
## Ada Boosting

from sklearn.ensemble import AdaBoostClassifier

ad_classifier = AdaBoostClassifier()

ad_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = ad_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Gradient Boosting

from sklearn.ensemble import GradientBoostingClassifier

gr_classifier = GradientBoostingClassifier()

gr_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = gr_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Xg Boosting

from xgboost import XGBClassifier

xg_classifier = XGBClassifier()

xg_classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = xg_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Xg Boosting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)
##Ensemble Voting Classifier

from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score

voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),

                                                  ('kn', kn_classifier),

                                                  ('svc_linear', svm_linear_classifier),

                                                  ('svc_rbf', svm_rbf_classifier),

                                                  ('gb', gb_classifier),

                                                  ('dt', dt_classifier),

                                                  ('rf', rf_classifier),

                                                  ('ad', ad_classifier),

                                                  ('gr', gr_classifier),

                                                  ('xg', xg_classifier),],

voting='soft')



for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,

            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,

            voting_classifier):

    clf.fit(X_train,y_train)

    y_pred = clf.predict(X_test)

    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))



# Predicting Test Set

y_pred = voting_classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)    
#The Best Classifier

print('The best classifier is:')

print('{}'.format(results.sort_values(by='Accuracy',ascending=False).head(5)))
#Applying K-fold validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator=lr_classifier, X=X_train, y=y_train,cv=10)

accuracies.mean()

accuracies.std()

print("Logistic Regression Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))