import pandas as pd

import numpy as np

import seaborn as sns; sns.set(color_codes=True)
one = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/1.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

one.drop('Sequence', axis = 1, inplace = True)



two = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/2.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

two.drop('Sequence', axis = 1, inplace = True)



three = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/3.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

three.drop('Sequence', axis = 1, inplace = True)



four = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/4.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

four.drop('Sequence', axis = 1, inplace = True)



five = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/5.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

five.drop('Sequence', axis = 1, inplace = True)



six = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/6.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

six.drop('Sequence', axis = 1, inplace = True)



seven = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/7.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

seven.drop('Sequence', axis = 1, inplace = True)



eight = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/8.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

eight.drop('Sequence', axis = 1, inplace = True)



nine = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/9.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

nine.drop('Sequence', axis = 1, inplace = True)



ten = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/10.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

ten.drop('Sequence', axis = 1, inplace = True)



eleven = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/11.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

eleven.drop('Sequence', axis = 1, inplace = True)



twelve = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/12.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

twelve.drop('Sequence', axis = 1, inplace = True)



thirteen = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/13.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

thirteen.drop('Sequence', axis = 1, inplace = True)



fourteen = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/14.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

fourteen.drop('Sequence', axis = 1, inplace = True)



fifteen = pd.read_csv('../input/activity-recognition/ActivityAccelerometer/15.csv', names=["Sequence", "x_acceleration", "y_acceleration", "z_acceleration","Labels"])

fifteen.drop('Sequence', axis = 1, inplace = True)
one['Person'] = 1

two['Person'] = 2

three['Person'] = 3

four['Person'] = 4

five['Person'] = 5

six['Person'] = 6

seven['Person'] = 7

eight['Person'] = 8

nine['Person'] = 9

ten['Person'] = 10

eleven['Person'] = 11

twelve['Person'] = 12

thirteen['Person'] = 13

fourteen['Person'] = 14

fifteen['Person'] = 15

frames = [one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve ,thirteen,fourteen , fifteen]



df= pd.concat(frames)



df.shape
df_rnd = df.sample(n=100000, random_state=6758)
df_rnd.dtypes
df_rnd.isna().sum()
df_rnd = df_rnd[df_rnd.Labels != 0] #Removing rows which had label zero 
df_rnd['Labels'].value_counts()
df_rnd.describe()
means = pd.DataFrame(columns = ['x_acceleration_mean','y_acceleration_mean','z_acceleration_mean','Labels'])

grouped = df.groupby(df.Labels)





lst = []

lst2 = []

lst3 = []

lst4 = []

for val in range(1,8):

    label = grouped.get_group(val)

    lst.append(label['x_acceleration'].mean())

    lst2.append(label['y_acceleration'].mean())

    lst3.append(label['z_acceleration'].mean())

    lst4.append(val)



means['x_acceleration_mean'] = lst

means['y_acceleration_mean'] = lst2

means['z_acceleration_mean'] = lst3

means['Labels'] = lst4



means
import matplotlib.pyplot as plt

%matplotlib inline 

%config InlineBackend.figure_format = 'retina'

plt.style.use("ggplot")
df_rnd.boxplot(column=['x_acceleration','y_acceleration','z_acceleration']);
from scipy import stats

temp = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
temp.boxplot(column=['x_acceleration','y_acceleration','z_acceleration']);
means = pd.DataFrame(columns = ['x_acceleration_mean','y_acceleration_mean','z_acceleration_mean','Labels'])

grouped = temp.groupby(temp.Labels)





lst = []

lst2 = []

lst3 = []

lst4 = []

for val in range(1,8):

    label = grouped.get_group(val)

    lst.append(label['x_acceleration'].mean())

    lst2.append(label['y_acceleration'].mean())

    lst3.append(label['z_acceleration'].mean())

    lst4.append(val)



means['x_acceleration_mean'] = lst

means['y_acceleration_mean'] = lst2

means['z_acceleration_mean'] = lst3

means['Labels'] = lst4



means
df_rnd['Labels'].value_counts().plot(kind='pie',autopct='%.2f')

plt.figlegend()

plt.show()
df_rnd.groupby('Labels')['x_acceleration'].plot.kde();

plt.autoscale(enable=True, axis= 'both',tight=None)

plt.figlegend();
sns.kdeplot(df_rnd['x_acceleration'],shade = True, color = 'blue')
df_rnd.groupby('Labels')['y_acceleration'].plot.kde();

plt.figlegend();
sns.kdeplot(df_rnd['y_acceleration'],shade = True, color = 'red')
df_rnd.groupby('Labels')['z_acceleration'].plot.kde();

plt.autoscale(enable=True, axis= 'both',tight=None)

plt.figlegend();
sns.kdeplot(df_rnd['z_acceleration'],shade = True, color = 'green')
from pandas.plotting import scatter_matrix

colors_palette = {1: 'red', 2:'green',3:'blue',4:'orange',5:'yellow',6:'cyan',7:'magenta'}

colors = [colors_palette[c] for c in df_rnd['Labels']]

scatter_matrix(df_rnd, alpha = 0.2, figsize = (16,16), diagonal = 'hist',c=colors)

plt.show()
df_rnd.corr(method='pearson')
sns.heatmap(df_rnd.corr(method='pearson'))
df['Labels'].value_counts()
Data = df_rnd.drop(columns = ['Person','Labels']).values

target = df_rnd['Labels'].values
from sklearn import preprocessing



target =  preprocessing.LabelEncoder().fit_transform(target)
np.unique(target, return_counts=True)
#Using standard scaling techniques



Data = preprocessing.StandardScaler().fit_transform(Data)
from sklearn.model_selection import train_test_split



D_train, D_test, t_train, t_test = train_test_split(Data,

                                                   target,

                                                   test_size = 0.3,

                                                   random_state = 6758)
from sklearn.neighbors import KNeighborsClassifier



knn_classifier = KNeighborsClassifier()

knn_classifier.fit(D_train, t_train)

kNN = knn_classifier.score(D_test, t_test)

print('KNN Classifier : ', kNN)
from sklearn.tree import DecisionTreeClassifier



dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state= 6758)

dt_classifier.fit(D_train, t_train)

dT = dt_classifier.score(D_test, t_test)

print('DecisionTreeClassifier : ', dT)
from sklearn.model_selection import RepeatedStratifiedKFold



cv_method = RepeatedStratifiedKFold(n_splits=5, 

                                    n_repeats=3, 

                                    random_state= 6758)
import numpy as np

params_KNN = {'n_neighbors': [1,3, 5, 7, 11, 15, 20, 25], 

              'p': [1, 2, 5]}
from sklearn.model_selection import GridSearchCV



gs_KNN = GridSearchCV(estimator=KNeighborsClassifier(), 

                      param_grid=params_KNN, 

                      cv=cv_method,

                      verbose=1,  # verbose: the higher, the more messages

                      scoring='accuracy',

                      n_jobs= -2,

                      return_train_score=True)
gs_KNN.fit(Data, target);
gs_KNN.best_params_
gs_KNN.best_score_
gs_KNN.cv_results_['mean_test_score']
import pandas as pd



results_KNN = pd.DataFrame(gs_KNN.cv_results_['params'])
results_KNN['test_score'] = gs_KNN.cv_results_['mean_test_score']
results_KNN['metric'] = results_KNN['p'].replace([1,2,5], ["Manhattan", "Euclidean", "Minkowski"])

results_KNN
import altair as alt

alt.renderers.enable('html')



alt.Chart(results_KNN, 

          title='KNN Performance Comparison'

         ).mark_line(point=True).encode(

    alt.X('n_neighbors', title='Number of Neighbors'),

    alt.Y('test_score', title='Mean CV Score', scale=alt.Scale(zero=False)),

    color='metric'

).interactive()
from sklearn.tree import DecisionTreeClassifier



df_classifier = DecisionTreeClassifier(random_state=6758)



params_DT = {'criterion': ['gini', 'entropy'],

             'max_depth': [5, 6, 7, 8, 10 , 12, 15, 17],

             'min_samples_split': [2, 3]}



gs_DT = GridSearchCV(estimator=df_classifier, 

                     param_grid=params_DT, 

                     cv=cv_method,

                     verbose=1,

                     n_jobs=-2,

                     scoring='accuracy')



gs_DT.fit(Data, target);
gs_DT.best_params_
gs_DT.best_score_
results_DT = pd.DataFrame(gs_DT.cv_results_['params'])

results_DT['test_score'] = gs_DT.cv_results_['mean_test_score']

results_DT.columns
alt.Chart(results_DT, 

          title='DT Performance Comparison'

         ).mark_line(point=True).encode(

    alt.X('max_depth', title='Maximum Depth'),

    alt.Y('test_score', title='Mean CV Score', aggregate='average', scale=alt.Scale(zero=False)),

    color='criterion'

).interactive()
t_pred_knn = gs_KNN.predict(D_test)
from sklearn import metrics

print('KNN Predicted accuracy score: ',metrics.accuracy_score(t_test, t_pred_knn))
t_pred_dt = gs_DT.predict(D_test)
print('DecisionTree accuracy score : ',metrics.accuracy_score(t_test,t_pred_dt))
# Classification report for KNN 



print(metrics.classification_report(t_test, t_pred_knn))
print(metrics.confusion_matrix(t_test, t_pred_knn))
#Classification report for Decision tree



print(metrics.classification_report(t_test, t_pred_dt, labels=np.unique(t_pred_dt)))
print(metrics.confusion_matrix(t_test, t_pred_dt))