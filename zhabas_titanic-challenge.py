# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.preprocessing import MaxAbsScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_women = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_train.head()
df_train.describe()
df_train.columns 
print(df_train.dtypes)
class_counts = df_train.groupby('Survived').size()

class_counts

df_train["Age"].head()
for c in df_train.columns :

    if df_train[c].dtype == np.float64 :

        fig = px.box(df_train[c], y=c)

        fig.show()
categorical_cols = ["Pclass","Sex","SibSp","Parch","Embarked"]

for c in categorical_cols : 

    print(df_train[c].value_counts().astype(np.float64))

    

    fig = go.Figure(data=[go.Pie(labels=df_train[c].value_counts().index, values=df_train[c].value_counts().values, hole=.3)])

    fig.update_layout(title = 'Distribution of the variable ' + str(c))

    fig.show()
percent_missing = df_train.isnull().sum() * 100 / len(df_train)

missing_value_df = pd.DataFrame({'column_name': df_train.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

missing_value_df
for i in range(1,4) :

    df_byclass = df_train[df_train["Pclass"]== i ]

    percent_missing = df_byclass.isnull().sum() * 100 / len(df_byclass)

    missing_value_df = pd.DataFrame({'column_name': df_byclass.columns,

                                     'percent_missing': percent_missing})

    missing_value_df.sort_values('percent_missing', inplace=True)

    print("For people who stayed in Class" + str(i))

    print(missing_value_df)

    
df_train.drop(columns = ["Cabin"], inplace = True)

df_train.dropna(subset = ["Embarked"],inplace = True)
tmp = df_train["Age"].mean()
df_train["Age"].fillna(value = tmp,inplace = True)
percent_missing = df_train.isnull().sum() * 100 / len(df_train)

missing_value_df = pd.DataFrame({'column_name': df_train.columns,

                                 'percent_missing': percent_missing})

missing_value_df.sort_values('percent_missing', inplace=True)

missing_value_df
df_train.drop(columns = ["Name"],inplace = True)

df_train.head()
df_copy = df_train.copy()

categorical_cols = df_copy.select_dtypes(exclude= 'number').columns.to_list()

le = preprocessing.LabelEncoder()

df_copy[categorical_cols] = df_copy[categorical_cols].apply(le.fit_transform)
#df_train = df_train.drop(columns = ["Name"])

correlation = df_copy.corr()

fig = go.Figure(data=go.Heatmap(z = correlation, x = df_copy.columns, y = df_copy.columns))

fig.show()
index = df_train["PassengerId"]

y = df_train["Survived"]

X = df_train.drop(columns = ["Survived","PassengerId"])
train,test = train_test_split(X, test_size=0.2)

y_train,y_test = train_test_split(y, test_size=0.2)
#One-hot encoder

ohc = preprocessing.OneHotEncoder(handle_unknown = "ignore")

X_train_encoded = ohc.fit_transform(train[categorical_cols])

X_test_encoded = ohc.transform(test[categorical_cols])



#Label encoder



encoder = preprocessing.LabelEncoder()

X_num = X

X_num[categorical_cols] = X[categorical_cols].apply(encoder.fit_transform)

train_num,test_num = train_test_split(X_num, test_size=0.2)
train_num.shape
scaler = MaxAbsScaler()

X_train = scaler.fit_transform(X_train_encoded)

X_test= scaler.transform(X_test_encoded)
#Use label encoding for KNN instead on one hot to avoid dimensionality curse



knn = KNeighborsClassifier(n_neighbors=10)





#Train the model using the training sets

knn.fit(X_train, y_train)



#Predict the response for test dataset

y_pred = knn.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
acc_val = [] #to store rmse values for different k

K_values = [k for k in range(50)]

for K in K_values:

    K = K+1

    model = KNeighborsClassifier(n_neighbors = K)



    model.fit(X_train, y_train)  #fit the model

    pred = model.predict(X_test) #make prediction on test set

    acc = metrics.accuracy_score(y_test, pred)

    

    acc_val.append(acc) #store accuracy values

    #print('Accuracy value for k= ' , K , 'is:', acc)
acc_df = pd.DataFrame(data = acc_val,index = K_values, columns = ["Accuracy"])

print(acc_df.head(5))

fig = px.line(acc_df, y="Accuracy", title='KNN accuracy evolution', line_shape="spline")

fig.show()
clf = LogisticRegression(random_state=0, solver='liblinear')

clf.fit(X_train, y_train)

clf.predict

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
clf = RandomForestClassifier(n_estimators = 10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

base_accuracy = metrics.accuracy_score(y_test, y_pred)

print("Accuracy:", base_accuracy)
def evaluate(model, test_features, test_labels):

    

    y_pred = clf.predict(test_features)

    accuracy = metrics.accuracy_score(y_test, y_pred)

    

    return accuracy


# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

# Create a based model

rf = RandomForestClassifier()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)



# Fit the grid search to the data

grid_search.fit(X_train, y_train)

print("Grid search best parameters")

print(grid_search.best_params_)

best_grid = grid_search.best_estimator_

grid_accuracy = evaluate(best_grid, X_test, y_test)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))
