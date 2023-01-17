# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





from sklearn.cross_validation import train_test_split

from sklearn import preprocessing

from sklearn.metrics import log_loss

from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",parse_dates=['Dates'])

test = pd.read_csv("../input/test.csv",parse_dates=['Dates'])
#Convert crime labels to numbers

le_crime = preprocessing.LabelEncoder()

crime = le_crime.fit_transform(train.Category)

 

#Get binarized weekdays, districts, and hours.

days = pd.get_dummies(train.DayOfWeek)

district = pd.get_dummies(train.PdDistrict)

hour = train.Dates.dt.hour

hour = pd.get_dummies(hour) 

 

#Build new array

train_data = pd.concat([hour, days, district], axis=1)

train_data['crime']=crime

 

#Repeat for test data

days = pd.get_dummies(test.DayOfWeek)

district = pd.get_dummies(test.PdDistrict)

 

hour = test.Dates.dt.hour

hour = pd.get_dummies(hour) 

 

test_data = pd.concat([hour, days, district], axis=1)

 

training, validation = train_test_split(train_data, train_size=.60)

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

 'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',

 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
model = BernoulliNB()

model.fit(training[features],training['crime'])

predicted = np.array(model.predict_proba(validation[features]))

log_loss(validation['crime'], predicted)
#Logistic Regression for comparison

model = LogisticRegression(C=.01)

model.fit(training[features], training['crime'])

predicted = np.array(model.predict_proba(validation[features]))

log_loss(validation['crime'], predicted) 
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',

'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

 

features2 = [x for x in range(0,24)]

features = features + features2
model = BernoulliNB()

model.fit(training[features],training['crime'])

predicted = np.array(model.predict_proba(validation[features]))

log_loss(validation['crime'], predicted)