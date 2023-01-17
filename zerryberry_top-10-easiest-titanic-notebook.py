# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd

# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df= pd.read_csv("../input/titanic/train.csv")

df_test= pd.read_csv("../input/titanic/test.csv")

df.head()
df['Died'] = 1 - df['Survived']



df.groupby('Sex').agg('mean')[['Survived', 'Died']].plot(kind='bar', figsize=(10,6), stacked=True);
figure = plt.figure(figsize=(25, 7))

plt.hist([df[df['Survived'] == 1]['Fare'], df[df['Survived'] == 0]['Fare']], 

         stacked=True, color = ['g','r'],

         bins = 50, label = ['Survived','Dead'])

plt.xlabel('Fare')

plt.ylabel('Number of passengers')

plt.legend();
## Finding out the missing values



df.isna().sum().sort_values(ascending = False)
df.corr()
df.drop({'Cabin', 'Age', 'Embarked'}, axis=1, inplace= True)



# from the test data set as well. 



df_test.drop({'Cabin', 'Age', 'Embarked'}, axis=1, inplace= True)
df_test.isna().sum().sort_values(ascending = False)
df_test[df_test.isna().T.any().T]
df_test1= pd.read_csv("../input/titanic/test.csv")



df_test1['Fare'].groupby([df_test1['Pclass'], df_test1['Sex']]).mean()
df_test1['Fare'].groupby([df_test1['Pclass'], df_test1['Sex']]).median()
# Setting up a loop to fill value for that specific row



for i in range(len(df_test['Fare'])):

    if df_test['PassengerId'][i] == 1044:

        df_test['Fare'][i] = 10

               
# Checking it..



df_test.iloc[[152]]['Fare']
print(df.shape)

print(df_test.shape)
df.head()
df_test.head()
id= df_test['PassengerId']



df.drop({'PassengerId', 'Died'}, axis=1, inplace= True)

df_test.drop({'PassengerId'}, axis=1, inplace= True)
print(df.shape)

print(df_test.shape)
y= df['Survived']

df.drop({'Survived'}, axis= 1, inplace= True)
df1= df

df2= df_test



df= pd.get_dummies(df)

df_test= pd.get_dummies(df_test)
for col in df.columns:

  if col not in df_test.columns:

    df.drop({col}, axis= 1, inplace= True)

    

for col in df_test.columns:

  if col not in df.columns:

    df_test.drop({col}, axis= 1, inplace= True)
# Checking out the shapes of both data sets:



print(df.shape)

print(df_test.shape)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier



# Splitting data for training, validation



X_train, X_test, y_train, y_test= train_test_split(df, y, random_state= 42)

# Training the model to do some geeky magic!

reg= GradientBoostingClassifier()

reg.fit(X_train, y_train)
#Some "scoring",  because.. why not?

reg.score(X_test, y_test)
# Here we are predicting the label for an out-of-the-sample data which is our prime objective 



a= reg.predict(df_test)
a= pd.DataFrame({'PassengerId': id, 'Survived': a})

a.set_index('PassengerId', inplace= True)



a.to_csv('titanic_new.csv')



# Then, setting the data in a way we are asked to- no index- so we can sort of make passenger_id our index







# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "titanic_new.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(a)
a