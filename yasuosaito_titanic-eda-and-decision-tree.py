# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 100) # Setting pandas to display a N number of columns

pd.set_option('display.max_rows', 10) # Setting pandas to display a N number rows

pd.set_option('display.width', 1000) # Setting pandas dataframe display width to N

from scipy import stats # statistical library

from statsmodels.stats.weightstats import ztest # statistical library for hypothesis testing

import plotly.graph_objs as go # interactive plotting library

import matplotlib.pyplot as plt # plotting library

import pandas_profiling # library for automatic EDA

%pip install autoviz # installing and importing autoviz, another library for automatic data visualization

from autoviz.AutoViz_Class import AutoViz_Class

from IPython.display import display # display from IPython.display

from itertools import cycle # function used for cycling over values



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print("")

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data and displaying some rows

df = pd.read_csv("/kaggle/input/titanic/train.csv")



display(df.head(10))
# The pandas profiling library is really useful on helping us understand the data we're working on.

# It saves us some precious time on the EDA process.

report = pandas_profiling.ProfileReport(df)

#display(report)
# Also, there is an option to generate an .HTML file containing all the information generated by the report.

report.to_file(output_file='report.html')
# Another great library for automatic EDA is AutoViz.

# With this library, several plots are generated with only 1 line of code.

# When combined with pandas_profiling, we obtain lots of information in a

# matter of seconds, using less then 5 lines of code.

AV = AutoViz_Class()



# Let's now visualize the plots generated by AutoViz.

report_2 = AV.AutoViz("/kaggle/input/titanic/train.csv", ",", "Survived")
# Installing and loading the library

!pip install dabl



import dabl
titanic_df = pd.read_csv('../input/titanic/train.csv')

titanic_df_clean = dabl.clean(titanic_df, verbose=1)
types = dabl.detect_types(titanic_df_clean)

print(types) 
dabl.plot(titanic_df, target_col="Survived")
ec = dabl.SimpleClassifier(random_state=0).fit(titanic_df, target_col="Survived") 
df.info()

# there are missing values and object(string)
df.isnull().sum()

# show count of missing value
# fill median to missing values of Age

df["Age"].fillna(df.Age.median(),inplace = True)



# remove Cabin

df = df.drop("Cabin", axis =1)
# number of appearances for each values in Embarked

df['Embarked'].value_counts()
# fill mode value to missing values of Embarked

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True) 
df.info()

# there are no missing values.

# there are object(string)
# remove PassengerId, Name and Ticket

df = df.drop("PassengerId", axis =1)

df = df.drop("Name", axis =1)

df = df.drop("Ticket", axis =1)

df.head()
# label-encode to Sex

df['Sex'].replace(['male', 'female'],[0,1], inplace =True)

df.head()
# extract column 'Embarked' for one-hot encoding

embarked = df['Embarked']

embarked
# crate one-hot dataframe for column 'Embarked'

embarked_one_hot = pd.get_dummies(embarked)

print(embarked_one_hot)
# drop column 'Embarked' and add it's one-hot encoded data

df = df.drop("Embarked", axis =1)

df = pd.concat([df, embarked_one_hot], axis=1)

df.head()
# split test data set

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier



X_train, X_test, y_train, y_test = train_test_split(

    df.iloc[:, 1:], df.iloc[:, 0], test_size = 0.20, random_state=1)

print('X_train shape: ',X_train.shape,' y_train shape: ', y_train.shape,' X_test shape: ', X_test.shape,' y_test shape: ', y_test.shape)
# ??????????????????????????????

model = DecisionTreeClassifier(criterion='gini', max_depth=5,random_state=0)



# ??????????????????

model.fit(X_train, y_train)
# predicted values

predict_y = model.predict(X_test)

predict_y
# correct values

import numpy as np

np.array(y_test)
# culclate accuracy

from sklearn.metrics import accuracy_score

accuracy_score(y_test, predict_y)
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
# extract passenger_id for submission

test_passenger_id = test_data.iloc[:, :1]

test_passenger_id
test_data.info()

# there are missing values and object(string)
test_data.isnull().sum()

# show count of missing value
# fill median to missing values of Age

test_data["Age"].fillna(test_data.Age.median(),inplace = True)



# remove Cabin

test_data = test_data.drop("Cabin", axis =1)



# fill median to missing values of Age

test_data["Fare"].fillna(test_data.Fare.median(),inplace = True)



# fill mode value to missing values of Embarked

test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True) 
test_data.info()

# there are no missing values.

# there are object(string)
# remove PassengerId, Name and Ticket

test_data = test_data.drop("PassengerId", axis =1)

test_data = test_data.drop("Name", axis =1)

test_data = test_data.drop("Ticket", axis =1)

test_data.head()
# label-encode to Sex

test_data['Sex'].replace(['male', 'female'],[0,1], inplace =True)

test_data.head()
# extract column 'Embarked' for one-hot encoding

embarked_test_data = test_data['Embarked']



# crate one-hot dataframe for column 'Embarked'

embarked_one_hot_test_data = pd.get_dummies(embarked_test_data)



# drop column 'Embarked' and add it's one-hot encoded data

test_data = test_data.drop("Embarked", axis =1)

test_data = pd.concat([test_data, embarked_one_hot_test_data], axis=1)

test_data.head()
test_data.info()

# there are no missing values and no object(string)
predictions = model.predict(test_data.values)

predictions
output = pd.DataFrame({'PassengerId': test_passenger_id.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")