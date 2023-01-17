import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

# ML algorithms;
# Algorithms
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
# Get train/test data
# Notice that train and test have same columns EXCEPT survial;
titanic_train = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/train.csv')
titanic_test = pd.read_csv('/kaggle/input/titanic-machine-learning-from-disaster/test.csv')
titanic_train.head(10)
titanic_test.head(10)
# Size of train data
titanic_train.shape

# Summary of numeric features; the count will tell if there are missing values;
titanic_train.describe()

# Info;
titanic_train.info()
# Function to check the missing percent of a DatFrame;
def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = round(df.isnull().sum().sort_values(ascending = False) * 100 /len(df),2)
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
# Lets check train and test data;
check_missing_data(titanic_train)
check_missing_data(titanic_test)
# Missing data: Cabin has high rate of missing data; insted of deleting the column,
# I will give 1 if Cabin is not null; otherwise 0;
titanic_train['Cabin']=np.where(titanic_train['Cabin'].isnull(),0,1)
titanic_test['Cabin']=np.where(titanic_test['Cabin'].isnull(),0,1)
# Combine train and test data, fill the missing values;
dataset = [titanic_train, titanic_test]

# def missing_data(x):
for data in dataset:
    #complete missing age with median
    data['Age'].fillna(data['Age'].mean(), inplace = True)

    #complete missing Embarked with Mode
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace = True)

        #complete missing Fare with median
    data['Fare'].fillna(data['Fare'].mean(), inplace = True)
check_missing_data(titanic_train)
check_missing_data(titanic_test)
# Delete the irrelavent columns: Name, Ticket (which is ticket code)
drop_column = ['Name','Ticket']
titanic_train.drop(drop_column, axis= 1, inplace = True)
titanic_test.drop(drop_column,axis = 1,inplace = True)
all_data = [titanic_train, titanic_test]

# Convert ‘Sex’ feature into numeric.
genders = {"male": 0, "female": 1}

for dataset in all_data:
    dataset['Sex'] = dataset['Sex'].map(genders)
titanic_train['Sex'].value_counts()
# Function of drawing graph;
def draw(graph):
    for p in graph.patches:
        height = p.get_height()
        graph.text(p.get_x()+p.get_width()/2., height + 5,height ,ha= "center")
# Draw survided vs. non-survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph= sns.countplot(x='Survived', hue="Survived", data=titanic_train)
draw(graph)
# Cabin and survived;
sns.set(style="darkgrid")
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Cabin", hue ="Survived", data = titanic_train)
draw(graph)
# Sex and survied;
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Sex", hue ="Survived", data = titanic_train)
draw(graph)
# Pclass and survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Pclass", hue ="Survived", data = titanic_train)
draw(graph)
# Embarked and survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Embarked", hue ="Survived", data = titanic_train)
draw(graph)
# We think embaked is not important, so drop it;
drop_column = ['Embarked']
titanic_train.drop(drop_column, axis=1, inplace = True)
titanic_test.drop(drop_column,axis=1,inplace=True)
# Parch vs survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Parch", hue ="Survived", data = titanic_train)
draw(graph)
# SibSp vs survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="SibSp", hue ="Survived", data = titanic_train)
draw(graph)
# Combine SibSp and Parch as new feature; 
# Combne train test first;
all_data=[titanic_train,titanic_test]

for dataset in all_data:
    dataset['Family'] = dataset['SibSp'] + dataset['Parch'] + 1
# Family vs survied
plt.figure(figsize = (8, 5))
graph  = sns.countplot(x ="Family", hue ="Survived", data = titanic_train)
draw(graph)
# Create bins of ages and check ages vs survived;
# Notice that different bins can be used;
# Add new column in all_data;
for dataset in all_data:
    dataset['Age_cat'] = pd.cut(dataset['Age'], bins=[0,12,20,40,120], labels=['Children','Teenage','Adult','Elder'])
    
plt.figure(figsize = (8, 5))
sns.barplot(x='Age_cat', y='Survived', data=titanic_train)
plt.figure(figsize = (8, 5))
ag = sns.countplot(x='Age_cat', hue='Survived', data=titanic_train)
draw(ag)
# Check fare vs survived;
# Create categorical of fare to plot fare vs Pclass first;
for dataset in all_data:
    dataset['Fare_cat'] = pd.cut(dataset['Fare'], bins=[0,10,50,100,550], labels=['Low_fare','median_fare','Average_fare','high_fare'])
plt.figure(figsize = (8, 5))
ag = sns.countplot(x='Pclass', hue='Fare_cat', data=titanic_train)
# Fare vs survived;
sns.barplot(x='Fare_cat', y='Survived', data=titanic_train)
# Use bin to convert ages to bins;
for dataset in all_data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 15, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 20), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 20) & (dataset['Age'] <= 26), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 26) & (dataset['Age'] <= 28), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 28) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 45), 'Age'] = 5
    dataset.loc[ dataset['Age'] > 45, 'Age'] = 6
titanic_train['Age'].value_counts()
# Remove features that are not sued, combined, etc
for dataset in all_data:
    drop_column = ['Age_cat','Fare','SibSp','Parch','Fare_cat','PassengerId']
    dataset.drop(drop_column, axis=1, inplace = True)
titanic_train.head()
# Correlation;
corr=titanic_train.corr()#['Survived']

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize = (12,8))
sns.heatmap(corr, 
            annot=True,
            mask = mask,
            cmap = 'RdBu',
            linewidths=.9, 
            linecolor='white',
            vmax = 0.3,
            fmt='.2f',
            center = 0,
            square=True)
plt.title("Correlations Matrix", y = 1,fontsize = 20, pad = 20);
# Re-organize the data; keep the columns with useful features;
input_cols = ['Pclass',"Sex","Age","Cabin","Family"]
output_cols = ["Survived"]
X_train = titanic_train[input_cols]
y_train = titanic_train[output_cols]

X_test = titanic_test
# Logistic regression;

model = LogisticRegression()
model.fit(X_train,y_train)
y_pred_lr=model.predict(X_test)
model.score(X_train,y_train)


from sklearn.model_selection import cross_val_score
-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
# KNN
model = KNeighborsClassifier(n_neighbors = 3) 
model.fit(X_train, y_train)  
y_pred_knn = model .predict(X_test)  
model.score(X_train,y_train)
# Gaussian naive bayesian
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
y_pred_gnb=model.predict(X_test) 
model.score(X_train,y_train)
# Linear SVM
model  = LinearSVC()
model.fit(X_train, y_train)

y_pred_svc = model.predict(X_test)
model.score(X_train,y_train)
# Random forest
model  = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred_rf = model.predict(X_test)
model.score(X_train,y_train)
# Decision tree
model = DecisionTreeClassifier() 
model.fit(X_train, y_train)
y_pred_dt = model.predict(X_test) 
model.score(X_train,y_train)