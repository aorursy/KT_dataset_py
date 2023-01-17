import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
# Load the train data

titanic_train = pd.read_csv('/kaggle/input/titanic/train.csv')

# Load the test data

titanic_test= pd.read_csv('/kaggle/input/titanic/test.csv')

test_result=pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

# Check the data

titanic_train
# Check the test data

titanic_test
titanic=[titanic_train,titanic_test]
# Descriptive statistics are very useful for initial exploration of the variables

# By default, only descriptives for the numerical variables are shown

# To include the categorical ones, you should specify this with an argument

titanic_train.describe(include='all')



# Note that categorical variables don't have some types of numerical descriptives

# and numerical variables don't have some types of categorical descriptives
titanic_test.describe(include='all')
#count rows and columns

titanic_train.shape
#No. of passengers died and survived

titanic_train.Survived.value_counts()
sns.countplot(titanic_train.Survived)
titanic_train.columns
titanic_train.isnull().sum()
titanic_test.isnull().sum()
#delete the cabin feature/column and others previously stated to exclude in train and test dataset

drop_column = ['PassengerId','Cabin', 'Ticket','Name']

for dataset in titanic:

    dataset.drop(drop_column, axis=1, inplace = True)
for dataset in titanic:

#complete missing age with median

    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)



#complete embarked with mode

    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)



#complete missing fare with median

    dataset['Fare'].fillna(dataset['Fare'].median(), inplace = True)
titanic_train.isnull().sum()
titanic_test.isnull().sum()
titanic_train.describe(include='all')
titanic_test.describe(include='all')
titanic_train.columns
# Visualize the count of passengers survived for the columns 'Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'



cols=['Pclass','Sex','SibSp','Parch']



n_rows=2

n_cols=2



fig,axs=plt.subplots(n_rows,n_cols,figsize=(n_rows*5,n_cols*5))

for r in range(n_rows):

    for c in range(n_cols):

        i=r*n_cols+c

        ax=axs[r][c]

        sns.countplot(titanic_train[cols[i]],hue=titanic_train.Survived,ax=ax)

        ax.legend(title='survived',loc='upper right')

plt.tight_layout
fig,axs=plt.subplots(1,1,figsize=(5,5))

sns.countplot(y=titanic_train['Embarked'],hue=titanic_train.Survived,ax=axs)
titanic_train.groupby('Sex')[['Survived']].mean()
titanic_train.pivot_table('Survived',index='Sex',columns='Pclass')
titanic_train.pivot_table('Survived',index='Sex',columns='Pclass').plot()
sns.barplot(x='Pclass',y='Survived',data=titanic_train)
age=pd.cut(titanic_train['Age'],[0,18,80])

titanic_train.pivot_table('Survived',['Sex',age],'Pclass')
sns.barplot(x=age,y='Survived',data=titanic_train)
Fare=pd.cut(titanic_train['Fare'],range(0,551,50))

titanic_train.pivot_table('Survived',['Sex',Fare],'Pclass')
sns.barplot(x=Fare,y='Survived',data=titanic_train)
Class=titanic_train.Pclass.map({1:'First',2:'Second',3:'Third'})

plt.scatter(titanic_train.Fare,Class,label='Passenger Paid')

plt.ylabel('Class')

plt.xlabel('Price')



plt.title('Price of each class')

plt.legend()

plt.show()
#Visualized the train data

#Final train data set

titanic_train.head()
titanic_train.shape
titanic_test.dtypes
#Labeling the object datas

from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for dataset in titanic:

#Encode the Sex column

    dataset.loc[:,'Sex']=labelencoder.fit_transform(dataset.loc[:,'Sex'].values)

#Encode the Embarked column

    dataset.loc[:,'Embarked']=labelencoder.fit_transform(dataset.loc[:,'Embarked'].values)
titanic_train
# The target(s) (dependent variable) is 'Survived'

Y_train = titanic_train['Survived']



# The inputs are everything BUT the dependent variable, so we can simply drop it

X_train = titanic_train.drop(['Survived'],axis=1)



X_test = titanic_test
#scale data

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train=sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
#create a fuction with many ml models

def models(X_train,Y_train):

    # use logistic regression model

    from sklearn.linear_model import LogisticRegression

    log=LogisticRegression(random_state=0)

    log.fit(X_train,Y_train)

    

    #use KNeighbors

    from sklearn.neighbors import KNeighborsClassifier

    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

    knn.fit(X_train,Y_train)

    

    #use SVC (linear kernel)

    from sklearn.svm import SVC

    svc_lin=SVC(kernel='linear',random_state=0)

    svc_lin.fit(X_train,Y_train)

    

    #use SVC (RBF kernel)

    from sklearn.svm import SVC

    svc_rbf=SVC(kernel='rbf',random_state=0)

    svc_rbf.fit(X_train,Y_train)

    

    #use GaussianNB

    from sklearn.naive_bayes import GaussianNB

    gauss=GaussianNB()

    gauss.fit(X_train,Y_train)

    

    #use Decision Tree

    from sklearn.tree import DecisionTreeClassifier

    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)

    tree.fit(X_train,Y_train)

    

    #use Random forest classifier

    from sklearn.ensemble import RandomForestClassifier

    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

    forest.fit(X_train,Y_train)

    

    

    #print the training accuracy

    print('[0]Logistic Regression Training Accuracy: ', log.score(X_train,Y_train))

    print('[1]KNeighbors Training Accuracy: ', knn.score(X_train,Y_train))

    print('[2]SVC Linear Training Accuracy: ', svc_lin.score(X_train,Y_train))

    print('[3]SVC RBF Training Accuracy: ', svc_rbf.score(X_train,Y_train))

    print('[4]Gaussian NB Training Accuracy: ', gauss.score(X_train,Y_train))

    print('[5]Decision Tree Training Accuracy: ', tree.score(X_train,Y_train))

    print('[6]Random Forest Training Accuracy: ', forest.score(X_train,Y_train))

    

    return log,knn,svc_lin,svc_rbf,gauss,tree,forest



    
#train all the models

model=models(X_train,Y_train)
Final_test=test_result.copy()
Final_test['Survived']=model[4].predict(X_test)
Final_test
Final_test.to_csv(r'C:\Users\trina\Titanic1\Final_test.csv',index=False)