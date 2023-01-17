import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree,svm
df = pd.read_csv('../input/StudentsPerformance.csv')
df.head()
df.describe()
df['math passed'] = df['math score'] > 50
df['reading passed'] = df['reading score'] > 50
df['writing passed'] = df['writing score'] > 50
df['all passed'] = df['math passed'] & df['reading passed'] & df['writing passed']
def plotPassedByColumn(column, df):
    fig = plt.figure(figsize=(10,4))
    plt.subplot(221)
    sns.countplot(x=column, hue='math passed', data=df)
    plt.subplot(222)
    sns.countplot(x=column, hue='reading passed', data=df)
    plt.subplot(223)
    sns.countplot(x=column, hue='writing passed', data=df)
    plt.subplot(224)
    sns.countplot(x=column, hue='all passed', data=df)
    
def barplotPercentage(column, df):
    fig = plt.figure(figsize=(10,4))
    plt.subplot(221)
    sns.barplot(x=column, y='math passed', data=df)
    plt.subplot(222)
    sns.barplot(x=column, y='reading passed', data=df)
    plt.subplot(223)
    sns.barplot(x=column, y='writing passed', data=df)
    plt.subplot(224)
    sns.barplot(x=column, y='all passed', data=df)
    
plotPassedByColumn('gender', df)
plotPassedByColumn('race/ethnicity', df)
result_types = ['math passed', 'reading passed', 'writing passed', 'all passed']
groups = ['group A', 'group B', 'group C', 'group D', 'group E']
result_type_performance = []
for group in groups:
    group_performance = [group]
    for result_type in result_types:
        values = df[(df[result_type]) & (df['race/ethnicity'] == group)].count() / df[df['race/ethnicity'] == group].count()
        group_performance.append(int(values[0].round(2) * 100))
    result_type_performance.append(group_performance)
#sns.barplot(x=groups, hue=groups, data=np.array(result_type_performance))
res_df = pd.DataFrame(result_type_performance)
res_df.columns = ['group', 'math passed', 'reading passed', 'writing passed', 'all passed']
barplotPercentage('group', res_df)
plotPassedByColumn('lunch', df)
plotPassedByColumn('test preparation course', df)
X = df[['gender', 'race/ethnicity', 'lunch', 'test preparation course']]
y = df['all passed']
encoder = OneHotEncoder(handle_unknown='ignore')
encoder.fit(X)
X = pd.get_dummies(X)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=.15,random_state=0)
scores = []
iterations = 100
for i in range(1,iterations + 1):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))
x = np.linspace(1, iterations, iterations)
results = pd.DataFrame({'n_neighbors':x, 'scores': scores})
argmax = np.argmax(scores)
(argmax, scores[argmax])
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train, y_train)
decision_tree.score(X_test, y_test)
lin_svm = svm.SVC(kernel='poly', gamma='scale')
lin_svm.fit(X_train, y_train)
lin_svm.score(X_test, y_test)