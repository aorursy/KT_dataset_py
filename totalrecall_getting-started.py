# Imports


# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
import xgboost as xgb


### set-up some custom functions for below
def get_year(date):
    if date == date: 
        return int(str(date)[:4])
    return date

def get_month(date):
    if date == date: 
        return int(str(date)[5:7])
    return date

def language_bucket(dataset):
    if dataset['language'] == 'en':
        val = 'en'
    else:
        val = 'non-en'
    return val

#### import the data
train_users  = pd.read_csv('../input/train_users.csv')
test_users    = pd.read_csv('../input/test_users.csv')
gender = pd.read_csv('../input/age_gender_bkts.csv')
sessions = pd.read_csv('../input/sessions.csv')
countries = pd.read_csv('../input/countries.csv')

all_users = pd.concat((train_users, test_users), axis=0, ignore_index=True)

##train_users[train_users['id'] == 'bibf93h56j']
##train_users['date_first_booking'].isnull()

##### clean the data some

train_users['age_range'] = pd.cut(train_users["age"], [0, 20, 40, 60, 80, 100])
train_users['signup_year']  = train_users['date_account_created'].apply(get_year)
train_users['language_bucket'] = train_users.apply(language_bucket, axis = 1)
train_users['booked'] = (train_users['country_destination'] != 'NDF').astype(int)

test_users['age_range'] = pd.cut(train_users["age"], [0, 20, 40, 60, 80, 100])
test_users['signup_year']  = train_users['date_account_created'].apply(get_year)
test_users['language_bucket'] = train_users.apply(language_bucket, axis = 1)
###test_users['booked']   = (test_users['country_destination'] != 'NDF').astype(int)


##### the age variable has a few missing values.. let's go ahead and put the average in for these
average_age  = train_users["age"].mean()

train_users["age"][np.isnan(train_users["age"])]   = average_age
test_users["age"][np.isnan(test_users["age"])]     = average_age
#### looking at the country distribution through a couple of variables 
fig, (axis1, axis2, axis3, axis4, axis5, axis6) = plt.subplots(6,1,figsize=(15,30))
sns.countplot(x='country_destination', data=train_users, palette="husl", ax=axis1)

sns.countplot(x='signup_flow', hue = "country_destination", data=train_users, palette="husl", ax=axis2)

sns.countplot(x='affiliate_channel', hue = "country_destination", data=train_users, palette="husl", ax=axis3)

sns.countplot(x='age_range', hue = "country_destination", data=train_users, palette="husl", ax=axis4)

sns.countplot(x='signup_year', hue = "country_destination", data=train_users, palette="husl", ax=axis5)

sns.countplot(x='language_bucket', hue = "country_destination", data=train_users, palette="husl", ax=axis6)

######## need to change the format of our variables so we can use the algo
# signup_method
train_users["signup_method"] = (train_users["signup_method"] == "basic").astype(int)
test_users["signup_method"]   = (test_users["signup_method"] == "basic").astype(int)

# signup_flow
train_users["signup_flow"] = (train_users["signup_flow"] == 3).astype(int)
test_users["signup_flow"]   = (test_users["signup_flow"] == 3).astype(int)

# language
train_users["language"] = (train_users["language"] == 'en').astype(int)
test_users["language"]   = (test_users["language"] == 'en').astype(int)

# affiliate_channel
train_users["affiliate_channel"] = (train_users["affiliate_channel"] == 'direct').astype(int)
test_users["affiliate_channel"]   = (test_users["affiliate_channel"] == 'direct').astype(int)

# affiliate_provider
train_users["affiliate_provider"] = (train_users["affiliate_provider"] == 'direct').astype(int)
test_users["affiliate_provider"]   = (test_users["affiliate_provider"] == 'direct').astype(int)
#### clense the data of non-numeric values 

from sklearn import preprocessing

for f in train_users.columns:
    if f == "country_destination" or f == "id": continue
    if train_users[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(np.unique(list(train_users[f].values) + list(test_users[f].values)))
        train_users[f] = lbl.transform(list(train_users[f].values))
        test_users[f]   = lbl.transform(list(test_users[f].values))
##In
# define training and testing sets

X_train = train_users.drop(["country_destination", "id", 'booked', 'age_range'],axis=1)
Y_train = train_users["country_destination"]
X_test  = test_users.drop(['id', 'age_range'],axis=1).copy()

##In
# modify country_destination to numerical values

country_num_dic = {'NDF': 0, 'US': 1, 'other': 2, 'FR': 3, 'IT': 4, 'GB': 5, 'ES': 6, 'CA': 7, 'DE': 8, 'NL': 9, 'AU': 10, 'PT': 11}
num_country_dic = {y:x for x,y in country_num_dic.items()}

Y_train    = Y_train.map(country_num_dic)

### Xgboost 

params = {"objective": "multi:softmax", "num_class": 12}

T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)

gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# convert type to integer
Y_pred = Y_pred.astype(int)

# change values back to original country symbols
Y_pred = Series(Y_pred).map(num_country_dic)
# Create submission

country_df = pd.DataFrame({
        "id": test_users["id"],
        "country": Y_pred
    })

submission = DataFrame(columns=["id", "country"])

# sort countries according to most probable destination country 
for key in country_df['country'].value_counts().index:
    submission = pd.concat([submission, country_df[country_df["country"] == key]], ignore_index=True)

####submission.to_csv('airbnb.csv', index=False)
##### add ndf to everyone
ndf_only = pd.DataFrame(test_users['id'])
ndf_only['country'] = 'NDF'

##submission_final = pd.concat([submission, ndf_only])
ndf_only.to_csv('airbnb.csv', index=False)
###### uh are the previous submissions formatted incorrectly or something?
######## checking via baseline submission
result = []
for index, row in test_users.iterrows():
    if isinstance(row['date_first_booking'], float):
        result.append([row['id'], 'NDF'])
        result.append([row['id'], 'US'])
        result.append([row['id'], 'other'])
        result.append([row['id'], 'FR'])
        result.append([row['id'], 'IT'])
    else:
        result.append([row['id'], 'US'])
        result.append([row['id'], 'other'])
        result.append([row['id'], 'FR'])
        result.append([row['id'], 'IT'])
        result.append([row['id'], 'GB'])
        
pd.DataFrame(result).to_csv('sub.csv', index = False, header = ['id', 'country'])
##result
results = pd.DataFrame(result)
results.columns = ['id', 'country']
results[results['id'] == 'qe9gwamyfk']
result


