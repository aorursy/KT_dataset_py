# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.

#print(check_output(["head", "../input/data.csv"]).decode("utf8"))
### Read in data
data = pd.read_csv("../input/data.csv")

### Remove features
data = data.drop("team_id", 1)
data = data.drop("team_name", 1)

data.columns
data[["loc_x", "loc_y", "lat", "lon"]]
data["playoffs"].value_counts()

data.describe()
mydata = data[["game_event_id", "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining", "period", "playoffs", "seconds_remaining", "shot_distance", "shot_made_flag", "shot_id"]] 
#print(data.head())
data['matchup'].value_counts()

def season_to_numeric(x):
    return int(x.split('-')[0]) - 1996

data["season"] = data["season"].apply(season_to_numeric)
#data["matchup"] = 
def home_or_away(x):
    if "@" in x:
        return -1
    elif "vs." in x:
        return 1
    else:
        return 0
data['matchup'] = data['matchup'].apply(home_or_away)
def time_from_start(x):
    if x["period"] <= 4:
        period_minutes = x["period"] * 12
    else:
        period_minutes = 48 + (x["period"] - 4) * 5
    return (period_minutes - x["minutes_remaining"]) * 60 - x["seconds_remaining"]
data['time_from_start'] = data.apply(time_from_start, axis = 1)
def categorical_to_freq(feature_name):
    my_counts = data[feature_name].value_counts()
    return data[feature_name].apply(lambda x: my_counts[x])
   
action_type = categorical_to_freq("action_type")
combined_shot_type = categorical_to_freq("combined_shot_type")
shot_type = categorical_to_freq("shot_type")
shot_zone_area = categorical_to_freq("shot_zone_area")
shot_zone_basic = categorical_to_freq("shot_zone_basic")
shot_zone_range = categorical_to_freq("shot_zone_range")



#my_counts = data['combined_shot_type'].value_counts()
#my_shot_type = data['combined_shot_type'].apply(lambda x: my_counts[x])
'''
### Feature engineering
shottype = pd.get_dummies(data["combined_shot_type"], prefix = "Shottype")
period = pd.get_dummies(data["period"], prefix = "Period")
season = pd.get_dummies(data["season"], prefix = "Season")
opponent = pd.get_dummies(data["opponent"], prefix = "Opponent")

# data["playoffs"], data["shot_distance"]
mydata = pd.concat([data['shot_made_flag'], data['shot_id'], shottype, period, season, data["playoffs"], data["shot_distance"], data[["loc_x", "loc_y", "lat", "lon", "playoffs"]], opponent], axis = 1)
'''

mydata = data[["matchup","season", "time_from_start", "game_event_id", "game_id", "lat", "loc_x", "loc_y", "lon", "minutes_remaining", "period", "playoffs", "seconds_remaining", "shot_distance", "shot_made_flag", "shot_id"]] 
mydata = pd.concat([mydata, action_type, combined_shot_type, shot_type, shot_zone_area, shot_zone_basic, shot_zone_range], axis = 1)


### Split mydata to train and test set 
train = mydata[mydata["shot_made_flag"].notnull()]
test = mydata[mydata["shot_made_flag"].isnull()]

target = train["shot_made_flag"]
train = train.drop(["shot_made_flag", "shot_id"], 1)
test_shot_id = test["shot_id"]
test = test.drop(["shot_made_flag", "shot_id"], 1)

'''
### Logistic Regression
lr = LogisticRegression().fit(train, target)
result = lr.predict_proba(test)

prob = result[: ,1]
'''



### GBRT
gbrt = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5,  random_state=0, loss='ls').fit(train, target)
gbrt_res = gbrt.predict(test)
output = pd.concat([test_shot_id.reset_index(drop = True), pd.Series(data = gbrt_res, name = 'shot_made_flag')], axis = 1)
#pd.Series(data = prob, name = "prob")
output.to_csv("my_third_submission.csv", index = False)
print(check_output(["cp", "../input/sample_submission.csv", "sample_submission.csv"]).decode("utf8"))
print(check_output(["head", "my_third_submission.csv"]).decode("utf8"))