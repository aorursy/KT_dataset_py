#Let's import everything we will need
import pandas as pd
import numpy as np
from IPython.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
import warnings
warnings.filterwarnings("ignore")
sns.set_style('whitegrid')

%matplotlib inline
#Read the csv file and create feature / label sets
X = pd.read_csv('../input/HR_comma_sep.csv')
X.head()

y = X['left']

print('Number of records: ', X.shape[0])
X.head()
X.isnull().values.ravel().sum()
X.dtypes
X.salary.unique()
X.salary.replace({'low':1,'medium':2,'high':3},inplace=True)
X.describe()
fig = plt.figure(figsize=(7,4))
corr = X.corr()
sns.heatmap(corr,annot=True,cmap='seismic',
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.title('Heatmap of Correlation Matrix')
plt.figure(figsize = (12,8))
plt.subplot(1,2,1)
plt.plot(X.satisfaction_level[X.left == 1],X.last_evaluation[X.left == 1],'ro', alpha = 0.2)
plt.ylabel('Last Evaluation')
plt.title('Employees who left')
plt.xlabel('Satisfaction level')

plt.subplot(1,2,2)
plt.title('Employees who stayed')
plt.plot(X.satisfaction_level[X.left == 0],X.last_evaluation[X.left == 0],'bo', alpha = 0.2,)
plt.xlim([0.4,1])
plt.ylabel('Last Evaluation')
plt.xlabel('Satisfaction level')
from sklearn.cluster import KMeans
kmeans_df =  X[X.left == 1].drop([ u'number_project',
       u'average_montly_hours', u'time_spend_company', u'Work_accident',
       u'left', u'promotion_last_5years', u'sales', u'salary'],axis = 1)
kmeans = KMeans(n_clusters = 3, random_state = 0).fit(kmeans_df)
print(kmeans.cluster_centers_)

left = X[X.left == 1]
left['label'] = kmeans.labels_
plt.figure(figsize=(10,7))
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation score')
plt.title('3 Clusters of employees who left')
plt.plot(left.satisfaction_level[left.label==0],left.last_evaluation[left.label==0],'o', alpha = 0.2, color = 'r')
plt.plot(left.satisfaction_level[left.label==1],left.last_evaluation[left.label==1],'o', alpha = 0.2, color = 'g')
plt.plot(left.satisfaction_level[left.label==2],left.last_evaluation[left.label==2],'o', alpha = 0.2, color = 'b')
plt.legend(['Winners','Frustrated','Bad Match'], loc = 'best', fontsize = 13, frameon=True)
winners = left[left.label ==0]
frustrated = left[left.label == 1]
bad_match = left[left.label == 2]

plt.figure(figsize=(10,4))
sns.kdeplot(winners.average_montly_hours, color = 'r', shade=True)
sns.kdeplot(bad_match.average_montly_hours, color ='b', shade=True)
sns.kdeplot(frustrated.average_montly_hours, color ='g', shade=True)
plt.legend(['Winners','Bad Match','Frustrated'])
plt.title('Leavers: Hours per month distribution')
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'satisfaction_level'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'satisfaction_level'] , color='r',shade=True, label='Left')
plt.title('Satisfaction levels')
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4),)
ax=sns.kdeplot(X.loc[(X['left'] == 0),'last_evaluation'] , color='b',shade=True,label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'last_evaluation'] , color='r',shade=True, label='Left')
plt.title('Last evaluation')
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'number_project'] , color='b',shade=True, label= 'Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'number_project'] , color='r',shade=True, label= 'Left')
plt.title('Number of projects')
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'average_montly_hours'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'average_montly_hours'] , color='r',shade=True, label='Left')
plt.title('Average monthly hours worked')
#produce kernel density estimate plots and histograms to look at each feature
fig = plt.figure(figsize=(10,4))
ax=sns.kdeplot(X.loc[(X['left'] == 0),'salary'] , color='b',shade=True, label='Stayed')
ax=sns.kdeplot(X.loc[(X['left'] == 1),'salary'] , color='r',shade=True, label='Left')
plt.title('Salary: (1-Low; 2-Medium; 3-high)')
fig = plt.figure(figsize=(10,4),)
sns.barplot(x = 'time_spend_company', y = 'left', data = X, saturation=1)
features = X[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','number_project']]
X = features
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, random_state=0,test_size=0.25)
print('Training set volume:', X_train.shape[0])
print('Test set volume:', X_test.shape[0])
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
accuracy_score(y_test,logreg.predict(X_test))
# Create Naive Bayes classifier
clf_gb = GaussianNB()
clf_gb.fit(X_train, y_train)
predicts_gb = clf_gb.predict(X_test)
print("GB Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predicts_gb))
#Create k-nn
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print("KNN5 Accuracy Rate, which is calculated by accuracy_score() is: %f" %accuracy_score(y_test,y_pred))
k_range = range(1,26)
scores=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))

plt.plot(k_range,scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing accuracy')
#Decision Tree
clf_dt = tree.DecisionTreeClassifier(min_samples_split=25)
clf_dt.fit(X_train, y_train)
predicts_dt = clf_dt.predict(X_test)
print("Decision tree Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predicts_dt))
#SVM -> takes a few seconds to run!
clf_svm = svm.SVC(kernel='rbf',probability=False)
clf_svm.fit(X_train,y_train)
predict_svm = clf_svm.predict(X_test)
print("SVM Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(y_test, predict_svm))
#Random forest classifier
clf_rf = RandomForestClassifier(n_estimators = 10,min_samples_split=2,max_depth=30)
clf_rf.fit(X_train, y_train)
accuracy_rf = clf_rf.score(X_test,y_test)
print("Random Forest Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_rf)
# The Random forest classifier seems to produce the best results so far, so let's try to optimise it!

max_depth = range(1,50)
scores=[]
for r in max_depth:
    clf_rf = RandomForestClassifier(n_estimators = 10,min_samples_split=2,max_depth=r)
    clf_rf.fit(X_train,y_train)
    y_pred=clf_rf.predict(X_test)
    scores.append(accuracy_score(y_test,y_pred))
    
plt.plot(max_depth,scores)
plt.xlabel('Value of r for Random Forest')
plt.ylabel('Testing accuracy')

# we're getting different values each time, but depth of around 30 seems to give good results.