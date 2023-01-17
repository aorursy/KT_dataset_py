import numpy as np

import pandas as pd

from datetime import datetime

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/ubersupplydemandgap/Uber Request Data.csv")

data.head(9)
data.info()
data1 = data[["Pickup point","Status",

              "Request timestamp","Drop timestamp"]]

data1.head(9)
def to_date(dat):

    dat = str(dat)

    if len(dat)>3:

        if "-" in dat:

            try:

                return datetime.strptime(dat,"%d-%m-%Y %H:%M:%S")

            except:

                return datetime.strptime(dat,"%d-%m-%Y %H:%M") 

        elif "/" in dat:

            try:

                return datetime.strptime(dat,"%d/%m/%Y %H:%M:%S")

            except:

                return datetime.strptime(dat,"%d/%m/%Y %H:%M") 

    else:

        return np.NaN
def tday(per):

    return per.strftime("%A")
timestamp1 = "04:00:00"

timestamp2 = "10:00:00"

timestamp3 = "16:00:00"

timestamp4 = "22:00:00"

t1 = datetime.strptime(timestamp1, "%H:%M:%S")

t2 = datetime.strptime(timestamp2, "%H:%M:%S")

t3 = datetime.strptime(timestamp3, "%H:%M:%S")

t4 = datetime.strptime(timestamp4, "%H:%M:%S")



def to_time(per):

    per = per.time()

    slot  = ""

    if per >=t1.time() and per <=t2.time():

        slot = "morning"

    

    elif per >t2.time() and per <=t3.time():

        slot = "daytime"

    

    elif per >t3.time() and per <=t4.time():

        slot = "evening"

    

    else:

        slot =  "midnight"

    return slot

        

def time_in_sec(per):

    try:

        

        return per.total_seconds()

    except:

        return pd.NaT

    
def tmonth(per):

    return per.strftime("%B")
data[data["Drop timestamp"]=='11/7/2016 13:00']["Drop timestamp"]="11/7/2016 13:00:00"
data1["Drop timestamp"] = data1["Drop timestamp"].apply(to_date)

data1["Request timestamp"] = data1["Request timestamp"].apply(to_date)

data1["time taken"] = data1["Drop timestamp"]- data1["Request timestamp"]

data1["total_time_in_sec"]=data1["time taken"].apply(time_in_sec)

data1["day of weak"]=data1["Request timestamp"].apply(tday)

data1["month"] = data1["Request timestamp"].apply(tmonth)

data1["time_slot"] = data1["Request timestamp"].apply(to_time)
data1.info()
data1.head()
sns.countplot(x ="Pickup point",data = data1)
sns.countplot(x = "time_slot",data= data1,hue = "Status")

sns.countplot(x = "Status",data= data1[data1["Pickup point"]=="City"])

sns.countplot(x = "Status",data= data1[data1["Pickup point"]!="City"])
a= []

for i in range(len(data1["Request timestamp"])):

    a.append(data1["Request timestamp"][i].hour)

sns.countplot(x =a ,hue=data1["Status"])
sns.countplot(x =a ,hue=data1["Pickup point"])
sns.countplot(x =data1["time_slot"] ,hue=data1["Pickup point"])

"""



   more evening requestes at airport

   more morning requeses at city



"""
ml_data = data1[["Pickup point","Status","total_time_in_sec","day of weak","time_slot"]]

ml_data.head()
X= ml_data.iloc[:,[0,3,4]]

X.head()
X = ml_data.iloc[:,[0,3,4]].values

y = ml_data.iloc[:, [1]].values





from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_0 = LabelEncoder()





X[:, 0] = labelencoder_0.fit_transform(X[:, 0])

X[:, 1] = labelencoder_0.fit_transform(X[:, 1])

X[:, 2] = labelencoder_0.fit_transform(X[:, 2])





onehotencoder = OneHotEncoder(categorical_features = [0,1,2])

X = onehotencoder.fit_transform(X).toarray()



pd.DataFrame(X).head()
X = pd.DataFrame(X[:,[0,2,3,4,5,7,8,9]]).values

pd.DataFrame(X).head()
y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)
y = ml_data.iloc[:, [1,0]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_y = LabelEncoder()

y[:,0] = labelencoder_y.fit_transform(y[:,0])



y[:,1] = labelencoder_y.fit_transform(y[:,1])

onehotencoder = OneHotEncoder(categorical_features = [0])

y = onehotencoder.fit_transform(y).toarray()

y
y=y[:,:-1]
y = y[:,:-1]
y
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(X, y)
def ml_array(per1,per2):

    

    day_dic={"Monday" : "0100","Tuesday" : "0001","Wednesday" : "0000","Thursday" : "0010","Friday" : "1000"}

    timeslot_dict = {"daytime" : "100","evening" : "010","morning" : "000","midnight" : "001"}

    predict_dict = {"00" : "trip will be completed","01":"no cars available","10":"cancled"}

    arr = []

    pre_arr=""

    arr.append(int(per2))

    for i in day_dic[tday(per1)]:

        arr.append(int(i))

    for i in timeslot_dict[to_time(per1)]:

        arr.append(int(i))

    temp = regressor.predict([arr])

    #predict_dict = {"00" : "trip will be completed","01":"no cars available","10":"cancled"}

    try:

        

        return predict_dict[str(int(temp[0][0]))+str(int(temp[0][1]))]

    except:

        return "enter proper date"

    
d = to_date(input("enter date\n format day-month-year hour:min:sec/\n\n"))

p = input("airport = 1\ncity = 0\n\n")

"""d="01-02-2016 12:22:00"

p="0"

"""

ml_array(d,p)

    