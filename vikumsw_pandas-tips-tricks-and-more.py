import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from pprint import pprint

print(os.listdir("../input"))
print(pd.__version__)

print(np.__version__)
import warnings

warnings.filterwarnings("ignore")
# Read & peek top

data = pd.read_csv("../input/titanic/train.csv")

data.head()
# Reading with Index column & peek tail

data2 = pd.read_csv("../input/titanic/train.csv",index_col='PassengerId')

data2.tail()
# Shape, Row Count, Column Count & Column Names

print('Shape of dataframe \t:', data.shape)

print('# of Rows \t\t:', data.shape[0])

print('# of Columns \t\t:', data.shape[1])

print('Columns in dataframe \t:', data.columns)
values = {}

arr = []

print('values is a ' ,type(values))

type(arr)
def getColumnsWithMissingValuesList(df):

    return [col for col in df.columns if df[col].isnull().any()] 



getColumnsWithMissingValuesList(data)
def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



cat_cols = getObjectColumnsList(data)

cat_cols
def getNumericColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]



num_cols = getNumericColumnsList(data)

num_cols
def getLowCardinalityColumnsList(df,cardinality):

    return [cname for cname in df.columns if df[cname].nunique() < cardinality and df[cname].dtype == "object"]



LowCardinalityColumns = getLowCardinalityColumnsList(data,10)

LowCardinalityColumns
data['Embarked'].nunique()
def PerformOneHotEncoding(df,columnsToEncode):

    return pd.get_dummies(df,columns = columnsToEncode)



oneHotEncoded_df = PerformOneHotEncoding(data,getLowCardinalityColumnsList(data,10))

oneHotEncoded_df.head()
# select only int64 & float64 columns

numeric_data = data.select_dtypes(include=['int64','float64'])



# select only object columns

categorical_data = data.select_dtypes(include='object')
numeric_data.head()
categorical_data.head()
def missingValuesInfo(df):

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100, 2)

    temp = pd.concat([total, percent], axis = 1,keys= ['Total', 'Percentage'])

    return temp.loc[(temp['Total'] > 0)]



missingValuesInfo(data)
# for Object columns fill using 'UNKOWN'

# for Numeric columns fill using median

def fillMissingValues(df):

    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    values = {}

    for a in cat_cols:

        values[a] = 'UNKOWN'



    for a in num_cols:

        values[a] = df[a].median()

        

    df.fillna(value=values,inplace=True)

    

    

HandleMissingValues(data)

data.head()
#check for NaN values

data.isnull().sum().sum()
# pass the DataFrame and percentage

def dropDataMissingColumns(df,percentage):

    print("Dropping columns where more than {}% values are Missing..".format(percentage))

    nan_percentage = df.isnull().sum().sort_values(ascending=False) / df.shape[0]

    missing_val = nan_percentage[nan_percentage > 0]

    to_drop = missing_val[missing_val > percentage/100].index.values

    df.drop(to_drop, axis=1, inplace=True)
def dropTargetMissingRows(df,target):

    print("Dropping Rows where Target is Missing..")

    df.dropna(axis=0, subset=[target], inplace=True)
def logistic(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

    lr=LogisticRegression()

    lr.fit(X_train,y_train)

    y_pre=lr.predict(X_test)

    print('Accuracy : ',accuracy_score(y_test,y_pre))
series = data['Fare']

d = {series.name : series}

df = pd.DataFrame(d) 

df.head()
PassengerClass = data['Pclass'].astype('category')

PassengerClass.describe()
# checks whether df contatins null values or object columns

def checkDataBeforeTraining(df):

    if(df.isnull().sum().sum() != 0):

        print("Error : Null Values Exist in Data")

        return False;

    

    if(len([cname for cname in df.columns if df[cname].dtype == "object"])>0):

        print("Error : Object Columns Exist in Data")

        return False;

    

    print("Data is Ready for Training")

    return True;
def getTrainX_TrainY(train_df,target):

    trainY = train_df.loc[:,target]

    trainX = train_df.drop(target, axis=1)

    return trainX,trainY
#impoting required libraries for demo

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=4, n_informative=2,random_state=0, shuffle=False)





# Subploting lets 2*2 figure with sizes (14*14)

f,ax=plt.subplots(2,2,figsize=(14,14))



#first plot

sns.scatterplot(x=X[:,0], y=y, ax=ax[0,0])

ax[0,0].set_xlabel('Feature 1 Values')

ax[0,0].set_ylabel('Y Values')

ax[0,0].set_title('Sactter Plot : Feature 1 vs Y')



#second plot

sns.scatterplot(x=X[:,1], y=y,ax=ax[0,1])

ax[0,1].set_xlabel('Feature 2 Values')

ax[0,1].set_ylabel('Y Values')

ax[0,1].set_title('Sactter Plot : Feature 2 vs Y')



#Third plot

sns.scatterplot(x=X[:,2], y=y,ax=ax[1,0])

ax[1,0].set_xlabel('Feature 3 Values')

ax[1,0].set_ylabel('Y Values')

ax[1,0].set_title('Sactter Plot : Feature 3 vs Y')



#Fourth plot

sns.scatterplot(x=X[:,3], y=y,ax=ax[1,1])

ax[1,1].set_xlabel('Feature 4 Values')

ax[1,1].set_ylabel('Y Values')

ax[1,1].set_title('Sactter Plot : Feature 4 vs Y')



plt.show()