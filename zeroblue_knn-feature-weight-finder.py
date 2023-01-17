#This code helps to find out feature multipliers for KNN.
#This is shown using some features derived by me but this method can be extended for other features as well.
#One needs to derive his own features and then apply similar approach to get the correct weights.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

recent_train = pd.read_csv("../input/train.csv")

#select a single x_y_grid at random
recent_train = recent_train[(recent_train["x"]>4.5) &(recent_train["x"]<5) &(recent_train["y"]>2) &(recent_train["y"]<2.3)]

#derive some features
def time_trig(max_time):
    time_array = np.linspace(0, 2*np.pi, max_time)
    sin_values = np.sin(time_array)
    cos_values = np.cos(time_array)
    return (sin_values, cos_values)
def feature_engineering(df):
    minute =(df["time"]//5)%288
    trig_arrays = time_trig(288)
    df['minute_sin'] = trig_arrays[0][minute]
    df['minute_cos'] = trig_arrays[1][minute]
    del minute
    day = (df['time']//1440)%365
    trig_arrays = time_trig(365)
    df['day_of_year_sin'] = trig_arrays[0][day]
    df['day_of_year_cos'] = trig_arrays[1][day]
    del day
    weekday = (df['time']//1440)%7
    trig_arrays = time_trig(7)
    df['weekday_sin'] = trig_arrays[0][weekday]
    df['weekday_cos'] = trig_arrays[1][weekday]
    del weekday
    df['month'] = (((df['time'])/43800)%12) * 2.4
    df['year'] = (df['time']//525600).astype(float)
    df.drop(['time'], axis=1, inplace=True)
    df['accuracy'] = np.log10(df['accuracy']).astype(float)
    return df
recent_train = feature_engineering(recent_train)

print("recent_train created")
#creating arbitrary test
test = recent_train.sample(axis = 0,frac = 0.05)
print ("selected_part and test created")
features = ['x', 'y', 'minute_sin', 'minute_cos', 'accuracy',
           'day_of_year_sin', 'day_of_year_cos', 
           'weekday_sin', 'weekday_cos', 'year']
#fw = np.ones(len(features))
fw = [ 1., 0.73704615, -0.04374113, -0.02376629, -0.02240783, -0.00821462,
  0.00900225, -0.00434318, -0.01833968, -0.00208674]
print (len(test))
colname = str(features)
test[colname] = list
index = iter(test.index)
test["done"] = 0
count = 0
final_index = test.tail(1).index.values[0]
for i in index:
    new_ld = pd.DataFrame(columns = features)
    for j in range(15):
        new_ld1 = abs(recent_train[features] - test.loc[i][features])
        new_ld1 = new_ld1.drop(i)
        new_ld1["target"] = (recent_train["place_id"] != test.loc[i]["place_id"]) + 0
        new_ld1["x+y"] = np.sum(new_ld1[features]*fw,axis = 1)#(new_ld1["x"])+(new_ld1["y"])#
        new_ld1 = new_ld1.sort_values("x+y")[0:50]
        count += 1
        if i != final_index:
            i = next(index)
        else:
            continue
        true = new_ld1[new_ld1["target"] == 0]
        false = new_ld1[new_ld1["target"] != 0]
        if (len(true)< 10) | (len(false)< 10):
            continue
        new_ld = new_ld.append(new_ld1)
    if len(new_ld) > 0:
        lr.fit(new_ld[features],new_ld["target"])
        test.set_value(i,colname,lr.coef_.ravel())
        test.set_value(i,"done",1)
        print ("current status: sample number",count) 
    else:
        print("Failed at", i)
#average or sum all the multipliers to get overall multiplier
actual_test2 = test[test["done"]==1]
final_weights = np.zeros(len(fw))
for lists in actual_test2[colname]:
    final_weights = final_weights + lists


print (features) 
print ("corresponding weights")
print (final_weights/final_weights[0])
