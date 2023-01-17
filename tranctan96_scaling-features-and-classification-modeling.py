import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
# Check if this data contains missing values

data.isnull().sum().max()
data['price_range'].value_counts()
data.describe().T
corr = data.corr()

plt.figure(figsize=(15,10))

sns.heatmap(corr, square=True, annot=True, annot_kws={'size':8})
sns.jointplot(data['ram'], data['price_range'],kind='kde')
sns.boxplot(data['price_range'], data['ram'])
X = data.drop(columns='price_range')

y = data['price_range']
X.var()
sns.distplot(X['m_dep'])
# Remove non-ordinal

X = X.drop(columns=['blue', 'dual_sim', 'four_g', 'three_g', 'touch_screen', 'wifi'])

# Remove colinearity

X = X.drop(columns=['fc', 'px_width', 'sc_w'])

# Remove low variance

X = X.drop(columns=['m_dep', 'clock_speed'])





# X = X[['ram']]

X.info()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import StandardScaler



# Normalize Training Data 

scaler = StandardScaler().fit(X_train)

X_train_std = scaler.transform(X_train)

X_test_std = scaler.transform(X_test)



#Converting numpy array to dataframe

X_train_std_df = pd.DataFrame(X_train_std, index=X_train.index, columns=X_train.columns)

X_test_std_df = pd.DataFrame(X_test_std, index=X_test.index, columns=X_test.columns) 
X_train_std_df.head()
train_std_data = pd.concat([X_train_std_df, y_train], axis=1)

train_std_data.var().sort_values()
plt.figure(figsize=(15,10))

sns.heatmap(train_std_data.corr(method='spearman'), annot=True)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
from sklearn.model_selection import cross_val_score
def plot_validation(param_grid, clf, X_train, y_train):

    val_error_rate = []



    for key in param_grid.keys():

        param_range = param_grid[key]

        for param in param_range:

            # https://stackoverflow.com/questions/337688/dynamic-keyword-arguments-in-python

            val_error = 1 - cross_val_score(clf.set_params(**{key: param}), X_train, y_train, cv=5).mean()

            val_error_rate.append(val_error)



        plt.figure(figsize=(15,7))

        plt.plot(param_range, val_error_rate, color='orange', linestyle='dashed', marker='o',

                 markerfacecolor='black', markersize=5, label='Validation Error')



        plt.xticks(np.arange(param_range.start, param_range.stop, param_range.step), rotation=60)

        plt.grid()

        plt.legend()

        plt.title('Validation Error vs. {}'.format(key))

        plt.xlabel(key)

        plt.ylabel('Validation Error')

        plt.show()

    



neighbors_range = range(1,200,5)

param_grid = {'n_neighbors': neighbors_range}

plot_validation(param_grid, knn, X_train_std_df, y_train)
best_k = 136



knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_std_df, y_train)

1-knn.score(X_test_std_df, y_test)
from sklearn.svm import SVC



svm = SVC(kernel='linear')
c_range =  range(1,200,20)

param_grid = {'C': c_range}

plot_validation(param_grid, svm, X_train_std_df, y_train)
best_c = 21

svm = SVC(kernel='linear',C=best_c)

svm.fit(X_train_std_df, y_train)

svm.score(X_test_std_df, y_test)
# Using GridSearchCV to tune hyperparameters

from sklearn.model_selection import GridSearchCV



param_grid = {'C': c_range,

              'gamma': [.1, .5, .10, .25, .50, 1]}

gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)

gs.fit(X_train_std_df,y_train)
print("The best hyperparameters {}.".format(gs.best_params_))

print("The Mean CV score of the best_estimator is {:.2f}.".format(gs.best_score_))
svm = SVC(kernel='rbf',C=1, gamma=0.1)

svm.fit(X_train_std_df, y_train)

svm.score(X_test_std_df, y_test)
from sklearn.metrics import classification_report,confusion_matrix



knn = KNeighborsClassifier(n_neighbors=best_k)

knn.fit(X_train_std_df, y_train)

pred = knn.predict(X_test_std_df)



print(knn.score(X_test_std_df,y_test))

print(classification_report(y_test,pred))



matrix=confusion_matrix(y_test,pred)

plt.figure(figsize = (10,7))

sns.heatmap(matrix,annot=True)
svm = SVC(kernel='linear',C=best_c)

svm.fit(X_train_std_df, y_train)

pred = svm.predict(X_test_std_df)



print(svm.score(X_test_std_df,y_test))

print(classification_report(y_test,pred))



matrix=confusion_matrix(y_test,pred)

plt.figure(figsize = (10,7))

sns.heatmap(matrix,annot=True,fmt=".2f")