import pandas as pd

age_gender = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\age_gender_bkts.csv")

age_gender.head()
# import pandas as pd

# df = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\countries.csv")

# df.head()
# import pandas as pd

# df = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\sample_submission_NDF.csv")

# df.head()
# import pandas as pd

# df = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\sessions.csv")

# df.head()
import pandas as pd

test = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\test_users.csv")

test.head()
import pandas as pd

train = pd.read_csv("C:\\School_class_data\\machine_learning\\airbnb_data\\train_users_2.csv")

train.head()
train.info()
test.info()
# cleaningn Missing data 
total = train.isnull().sum().sort_values(ascending = False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis=1, keys=['total', 'percent'])

missing_data.head(20)
data = train.append(test)

data

data.reset_index(inplace = True, drop = True)
import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline

sns.countplot(data['age'])
%matplotlib inline

sns.countplot(data['first_affiliate_tracked'])
%matplotlib inline

sns.countplot(data['date_first_booking'])
data.head()
print(data.isnull().sum())
import numpy as np
data.age.replace('-unknown-', np.nan, inplace=True)

data.country_destination.replace('-unknown-', np.nan, inplace=True)

data.date_first_booking.replace('-unknown-', np.nan, inplace=True)

data.first_affiliate_tracked.replace('-unknown-', np.nan, inplace=True)
data.loc[data.age > 100, 'age'] = np.nan

data.loc[data.age < 18, 'age'] = np.nan
data.head()
data.groupby('gender').age.agg(['min','max','mean','count'])
data.groupby('gender').age.mean().plot(kind='bar')
data.gender.value_counts(dropna=False).plot(kind='bar')
train.country_destination.value_counts(normalize=True).plot(kind='bar',title='Countries Visited by AirBNB Users')
data.language.value_counts(sort=True)
print(data.isnull().sum())
data
# Preprocessing Data
import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt 

from sklearn import preprocessing 



names = [

    'affiliate_channel','affiliate_provider' ,'country_destination',

    'first_affiliate_tracked' ,'first_browser','first_device_type','gender','language',

    'signup_app','signup_method'  

]



data_dummies = pd.get_dummies(data,columns = names)



# data_dummies.describe()

# data.dummies.info()

# data.dummies.head()
data_dummies.describe()
data_dummies.head()
# from sklearn.model_selection import train_test_split

# from sklearn.metrics import accuracy_score

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.tree import DecisionTreeClassifier
# df_train_target = df_train['country_destination']

# df_train = df_train.drop(['country_destination'],1)

# df_test_target = df_test['country_destination']

# df_test = df_test.drop(['country_destination'],1)
# from sklearn.ensemble import RandomForestClassifier



# forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=123456)

# forest.fit(df_train, df_train_target)
# from sklearn.metrics import accuracy_score



# predicted = forest.predict(df_test)

# accuracy = accuracy_score(df_test_target, predicted)



# print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')

# print(f'Mean accuracy score: {accuracy:.3}')