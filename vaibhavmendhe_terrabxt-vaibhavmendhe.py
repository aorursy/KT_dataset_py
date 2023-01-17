# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

train_data = pd.read_csv("../input/into-the-future/train.csv",index_col='time',parse_dates=True)
train_data.head()
train_data.info()
dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
train_data = pd.read_csv("../input/into-the-future/train.csv", parse_dates=['time'], date_parser=dateparse)
#converting the date-time object into float object

def datetime_to_float(d):
    return d.timestamp()

for i in range(len(train_data["time"])):
    train_data.loc[i, "time"]=datetime_to_float(train_data.loc[i, "time"])
x=train_data.loc[:, ["time","feature_1"]] #featuers
y=train_data["feature_2"] #labels
#creating an object of linear regression model and trainng it on training data


reg = RandomForestRegressor(n_estimators = 50, random_state = 0) 
reg.fit(x, y)
#loading test data into a new dataframe

test_Data=pd.read_csv("../input/into-the-future/test.csv")
test_Data.head()
dateparse = lambda dates: pd.datetime.strptime(dates, "%Y-%m-%d %H:%M:%S")
test_Data = pd.read_csv("../input/into-the-future/test.csv", parse_dates=['time'], date_parser=dateparse)
for i in range(len(test_Data["time"])):
    test_Data.loc[i, "time"]=datetime_to_float(test_Data.loc[i, "time"])
test_Data.head()
x_test=test_Data.loc[:, ["time","feature_1"]]
ids=test_Data["id"]
#predicting the "feature_2" value using the Multiple linear regression model created above

y_pred=reg.predict(x_test)
y_pred
test_result=pd.DataFrame({"id": ids, "feature_2":y_pred})
test_result
test_result.to_csv("/kaggle/working/solution.csv", index=False)
# My Name: Vaibhav Ashok Mendhe
# Mail id: iamvaibhav6@gmail.com
# Contact No: 9145257767/8208853981