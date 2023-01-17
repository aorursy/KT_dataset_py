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
data_dictionary=pd.read_csv("/kaggle/input/exam-for-students20200527/data_dictionary.csv")

train_data=pd.read_csv("/kaggle/input/exam-for-students20200527/train.csv")

station_info=pd.read_csv("/kaggle/input/exam-for-students20200527/station_info.csv")

test_data=pd.read_csv("/kaggle/input/exam-for-students20200527/test.csv")

sample_submission=pd.read_csv("/kaggle/input/exam-for-students20200527/sample_submission.csv")

city_info=pd.read_csv("/kaggle/input/exam-for-students20200527/city_info.csv")



test_data
train_data.isnull().any()
train_data=train_data[(train_data.TimeToNearestStation.isnull()==False) & (train_data.MinTimeToNearestStation.isnull()==False)

           & (train_data.MaxTimeToNearestStation.isnull()==False)]

train_data.isnull().any()
temp=pd.DataFrame(train_data["Prefecture"].unique())

train_data["inaka"]=0

train_data["chuken"]=0

train_data.loc[train_data["Prefecture"] == "Kanagawa Prefecture", 'chuken'] = 1

train_data.loc[train_data["Prefecture"] == "Chiba Prefecture", 'chuken'] = 1

train_data.loc[train_data["Prefecture"] == "Ibaraki Prefecture", 'inaka'] = 1

train_data.loc[train_data["Prefecture"] == "Gunma Prefecture", 'inaka'] = 1

train_data.loc[train_data["Prefecture"] == "Tochigi Prefecture", 'inaka'] = 1





    

test_data["inaka"]=0

test_data["chuken"]=1



temp=pd.DataFrame(train_data["CityPlanning"].unique())



for i in range(len(temp)):

    temp_row=temp.loc[i,0]

    if temp_row:

        train_data[temp_row]=train_data.CityPlanning.where(train_data.CityPlanning == temp_row,0).replace(temp_row,1)

        test_data[temp_row]=test_data.CityPlanning.where(test_data.CityPlanning == temp_row,0).replace(temp_row,1)



train_data.isnull().any()

train_data.isnull().any()
tochigara=list(train_data.columns[36:50])

tochigara.append(train_data.columns[52])

tochigara.append("MinTimeToNearestStation")

tochigara.append("MaxTimeToNearestStation")

tochigara.append("inaka")

tochigara.append("chuken")

tochigara.append("BuildingYear")





tochigara
means=pd.DataFrame(train_data.groupby(['Municipality']).median())

name=list(means.index)

means["BuildingYear"][1]

for i in range(len(name)):

    train_data.loc[(train_data.Municipality == str(name[i]))&(train_data.BuildingYear.isnull()),"BuildingYear"] =means["BuildingYear"][i]



                    
train_data.isnull().any()
means=pd.DataFrame(test_data.groupby(['Municipality']).median())

name=list(means.index)

for i in range(len(name)):

    test_data.loc[(test_data.Municipality == str(name[i]))&(test_data.BuildingYear.isnull()),"BuildingYear"] =means["BuildingYear"][i]

test_data.isnull().any()
make_train_data_X=train_data[tochigara]

make_train_data_Y=train_data["TradePrice"]

make_test_data=test_data[tochigara]

make_test_data["MinTimeToNearestStation"] = make_test_data["MinTimeToNearestStation"].fillna(make_test_data["MinTimeToNearestStation"].median())

make_test_data["MaxTimeToNearestStation"] = make_test_data["MaxTimeToNearestStation"].fillna(make_test_data["MaxTimeToNearestStation"].median())

make_train_data_X["BuildingYear"] = make_train_data_X["BuildingYear"].fillna(make_train_data_X["BuildingYear"].median())

make_train_data_X.isnull().any()
from sklearn.ensemble import RandomForestRegressor

clf = RandomForestRegressor(random_state=0)

clf = clf.fit(make_train_data_X, make_train_data_Y)

pred = clf.predict(make_test_data)

test_data["TradePrice"]=pd.DataFrame(pred).astype(int)

submit=test_data[["id","TradePrice"]]

submit.to_csv("submit.csv",index=False)
pred