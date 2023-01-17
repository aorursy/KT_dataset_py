import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing

#　Load data
train = pd.read_csv('../input/train.csv')
test =  pd.read_csv('../input/test.csv')
test.columns
### Combine to data
# data = pd.concat([train, test], axis = 1)

### Data cleaning
train_cl = train.copy()

### Just take first 100 samples for bootstraping test
#train_cl = train_cl.iloc[:100]
dow = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

train_cl["DayOfWeek"] = train_cl.DayOfWeek.map(dow)
### Split Dates to multiple features, do not need minutes and seconds here, drop "Dates" features later
mydt = pd.to_datetime(train_cl.Dates)
train_cl["Year"] = mydt.dt.year
train_cl["Month"] = mydt.dt.month
train_cl["Day"] = mydt.dt.day
train_cl["Hour"] = mydt.dt.hour
train_cl.drop("Dates", axis = 1, inplace = True)
### Feature 'Descript' and 'Address' dropped
train_cl.drop("Descript", axis = 1, inplace = True)
train_cl.drop("Address", axis = 1, inplace = True)
### numeric: X, Y; categorical: others
train_cl.columns
#for feat in ["Year", "Month", "Day", "Hour", "DayOfWeek"]:
for feat in ['Category', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Year', 'Month', 'Day', 'Hour']:
### Print value_counts for all features
#for feat in train_cl.columns:
    print("***Value_counts for feature: {} \n{} \n".format(feat, train_cl[feat].value_counts()))
#### should get DataFrame data

# data_cl = data.copy()

data_cl = data[["X", "Y"]]

feat_list = ['Category', 'DayOfWeek', 'PdDistrict', 'Resolution', 'Year', 'Month', 'Day', 'Hour']
for feat in feat_list:
    dummies = pd.get_dummies(data_cl[feat])
    dummies = dummies.add_prefix("{}#".format(feat))
    data_cl = data_cl.join(dummies)
