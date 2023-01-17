import pandas as pd



data = pd.read_csv('../input/train.csv')

data.shape

data.head()

data.describe()

data['Age'].fillna(data['Age'].median(), inplace=True)

data.describe()

survived_sex = data[data['Survived']==1]['Sex'].value_counts()

dead_sex = data[data['Survived']==0]['Sex'].value_counts()

survived_sex , dead_sex
