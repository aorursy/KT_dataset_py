# Packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import seaborn as sns

import random

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import LinearSVC 

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier
# Data

data = pd.read_csv('../input/KaggleV2-May-2016.csv')

print(data.head())
# Counting Nans

data.isnull().sum()
# Rename columns

data.rename(columns = {'Hipertension':'Hypertension', 'Handcap':'Handicap'}, inplace = True)

print(data.columns)
data.info()
# Gender description

data["Gender"].describe()
# Changing types for Dates

data.ScheduledDay = data.ScheduledDay.apply(lambda x: np.datetime64(x, 'D'))

data.AppointmentDay = data.AppointmentDay.apply(lambda x: np.datetime64(x, 'D'))



# Checking alterations

data.info()
# Age description

data["Age"].describe()
# Age Boxplot

fig_age, ax1 = plt.subplots()

ax1.set_title('Age')

ax1.boxplot(data["Age"])
# Selecting good data

data = data[(data["Age"] >= 0)]



# Age Boxplot after changes

fig_age, ax1 = plt.subplots()

ax1.set_title('Age')

ax1.boxplot(data["Age"])
# Neighbourhood description

data["Neighbourhood"].describe()
# Neighbourhood description

data["Neighbourhood"].unique()
# Changing type

data['Scholarship'] = data['Scholarship'].astype(object)

data['Hypertension'] = data['Hypertension'].astype(object)

data['Diabetes'] = data['Diabetes'].astype(object)

data['Alcoholism'] = data['Alcoholism'].astype(object)

data['Handicap'] = data['Handicap'].astype(object)

data['SMS_received'] = data['SMS_received'].astype(object)



# Checking alterations

data.info()
# Scholarship description

data["Scholarship"].describe()
# Hypertension description

data["Hypertension"].describe()
# Diabetes description

data["Diabetes"].describe()
# Alcoholism description

data["Alcoholism"].describe()
# Handicap We don't see any problem with the Hypertension.ap description

data["Handicap"].describe()
# Handicap description

data["Handicap"].unique()
# SMS_received description

data["SMS_received"].describe()
# No-show description

data["No-show"].describe()
# Creating waiting time between Schedulment and Appointment

data["Waiting"] = (data.AppointmentDay - data.ScheduledDay).dt.days



# Getting week day for each date

data['Weekday_scheduled'] = data.ScheduledDay.apply(lambda x: x.weekday())

data['Weekday_Appointment'] = data.AppointmentDay.apply(lambda x: x.weekday())



# Encoding for days

def weekday_encoding(series):

    if series == 0:

        return "Lundi"

    elif series == 1:

        return "Mardi"

    elif series == 2:

        return "Mercredi"

    elif series == 3:

        return "Jeudi"

    elif series == 4:

        return "Vendredi"   

    elif series == 5:

        return "Samedi"

    elif series == 6:

        return "Dimanche"

    

# Encoding results

data['Weekday_scheduled'] = data['Weekday_scheduled'].apply(weekday_encoding)

data['Weekday_Appointment'] = data['Weekday_Appointment'].apply(weekday_encoding)
# Frequencies for Gender

data["Gender"].value_counts()
# Representation

plt.bar(["F","M"],data["Gender"].value_counts())

plt.ylabel('Frequency')

plt.title('Gender')

plt.show()
# Frequencies for No-show by Gender

pd.crosstab(data["Gender"],data["No-show"])
# Representation

p1 = plt.bar(["F","M"], pd.crosstab(data["Gender"],data["No-show"])["No"])

p2 = plt.bar(["F","M"], pd.crosstab(data["Gender"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Gender"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by gender')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Age description

data["Age"].describe()
# Age Boxplot after changes

fig_age, ax1 = plt.subplots()

ax1.set_title('Age')

ax1.boxplot(data["Age"])
# Histogram

plt.hist(data["Age"], rwidth=0.8, bins=20)
# Age Boxplot by No-show

fig_age, ax1 = plt.subplots()

ax1.set_title('No-show by Age')

ax1.boxplot([data[data["No-show"]=="Yes"]["Age"],data[data["No-show"]=="No"]["Age"]])

plt.gca().xaxis.set_ticklabels(['No-show : Yes', 'No-show : No'])
# Frequencies for Neighbourhood

data["Neighbourhood"].value_counts()
# Representation

plt.bar(data["Neighbourhood"].value_counts().index,data["Neighbourhood"].value_counts())

plt.ylabel('Frequency')

plt.title('Neighbourhood')

plt.show()
# Frequencies for No-show by Neighbourhood

pd.crosstab(data["Neighbourhood"],data["No-show"])
# Representation

p1 = plt.bar(np.unique(data["Neighbourhood"]), pd.crosstab(data["Neighbourhood"],data["No-show"])["No"])

p2 = plt.bar(np.unique(data["Neighbourhood"]), pd.crosstab(data["Neighbourhood"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Neighbourhood"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Neighbourhood')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Scholarship

data["Scholarship"].value_counts()
# Representation

plt.bar(["No","Yes"],data["Scholarship"].value_counts())

plt.ylabel('Frequency')

plt.title('Scholarship')

plt.show()
# Frequencies for No-show by Scholarship

pd.crosstab(data["Scholarship"],data["No-show"])
# Representation

p1 = plt.bar(["No Scholarship","Scholarship"], pd.crosstab(data["Scholarship"],data["No-show"])["No"])

p2 = plt.bar(["No Scholarship","Scholarship"], pd.crosstab(data["Scholarship"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Scholarship"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Scholarship')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Hypertension

data["Hypertension"].value_counts()
# Representation

plt.bar(["No","Yes"],data["Hypertension"].value_counts())

plt.ylabel('Frequency')

plt.title('Hypertension')

plt.show()
# Frequencies for No-show by Hypertension

pd.crosstab(data["Hypertension"],data["No-show"])
# Representation

p1 = plt.bar(["No Hypertension","Hypertension"], pd.crosstab(data["Hypertension"],data["No-show"])["No"])

p2 = plt.bar(["No Hypertension","Hypertension"], pd.crosstab(data["Hypertension"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Hypertension"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Hypertension')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Diabetes

data["Diabetes"].value_counts()
# Representation

plt.bar(["No","Yes"],data["Diabetes"].value_counts())

plt.ylabel('Frequency')

plt.title('Diabetes')

plt.show()
# Frequencies for No-show by Diabetes

pd.crosstab(data["Diabetes"],data["No-show"])
# Representation

p1 = plt.bar(["No Diabetes","Diabetes"], pd.crosstab(data["Diabetes"],data["No-show"])["No"])

p2 = plt.bar(["No Diabetes","Diabetes"], pd.crosstab(data["Diabetes"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Diabetes"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Diabetes')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Alcoholism

data["Alcoholism"].value_counts()
# Representation

plt.bar(["No","Yes"],data["Alcoholism"].value_counts())

plt.ylabel('Frequency')

plt.title('Alcoholism')

plt.show()
# Frequencies for No-show by Alcoholism

pd.crosstab(data["Alcoholism"],data["No-show"])
# Representation

p1 = plt.bar(["No Alcoholism","Alcoholism"], pd.crosstab(data["Alcoholism"],data["No-show"])["No"])

p2 = plt.bar(["No Alcoholism","Alcoholism"], pd.crosstab(data["Alcoholism"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Alcoholism"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Alcoholism')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Handicap

data["Handicap"].value_counts()
# Representation

plt.bar(range(len(data["Handicap"].value_counts())),data["Handicap"].value_counts())

plt.ylabel('Frequency')

plt.title('Handicap')

plt.show()
# Frequencies for No-show by Handicap

pd.crosstab(data["Handicap"],data["No-show"])
# Representation

p1 = plt.bar(range(len(data["Handicap"].value_counts())), pd.crosstab(data["Handicap"],data["No-show"])["No"])

p2 = plt.bar(range(len(data["Handicap"].value_counts())), pd.crosstab(data["Handicap"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["Handicap"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by Handicap')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for SMS_received

data["SMS_received"].value_counts()
# Representation

plt.bar(["No","Yes"],data["SMS_received"].value_counts())

plt.ylabel('Frequency')

plt.title('SMS_received')

plt.show()
# Frequencies for No-show by SMS_received

pd.crosstab(data["SMS_received"],data["No-show"])
# Representation

p1 = plt.bar(["No SMS received","SMS received"], pd.crosstab(data["SMS_received"],data["No-show"])["No"])

p2 = plt.bar(["No SMS received","SMS received"], pd.crosstab(data["SMS_received"],data["No-show"])["Yes"],

             bottom=pd.crosstab(data["SMS_received"],data["No-show"])["No"])



plt.ylabel('Frequency')

plt.title('No-show by SMS_received')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for Weekday Appointment

data["Weekday_Appointment"].value_counts()
# Representation

plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],[data["Weekday_Appointment"].value_counts()["Lundi"],data["Weekday_Appointment"].value_counts()["Mardi"],data["Weekday_Appointment"].value_counts()["Mercredi"],data["Weekday_Appointment"].value_counts()["Jeudi"],data["Weekday_Appointment"].value_counts()["Vendredi"],data["Weekday_Appointment"].value_counts()["Samedi"],0])

plt.ylabel('Frequency')

plt.title('Weekday_Appointment')

plt.show()
# Frequencies for No-show by Weekday_Appointment

pd.crosstab(data["Weekday_Appointment"],data["No-show"])
# Representation

p1 = plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"], [pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Lundi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Mardi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Mercredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Jeudi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Vendredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Samedi"],0])

p2 = plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"], [pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Lundi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Mardi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Mercredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Jeudi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Vendredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["Yes"]["Samedi"],0],

             bottom=[pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Lundi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Mardi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Mercredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Jeudi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Vendredi"],pd.crosstab(data["Weekday_Appointment"],data["No-show"])["No"]["Samedi"],0])



plt.ylabel('Frequency')

plt.title('No-show by Weekday_Appointment')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Frequencies for WWeekday scheduled

data["Weekday_scheduled"].value_counts()
# Representation

plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"],[data["Weekday_scheduled"].value_counts()["Lundi"],data["Weekday_scheduled"].value_counts()["Mardi"],data["Weekday_scheduled"].value_counts()["Mercredi"],data["Weekday_scheduled"].value_counts()["Jeudi"],data["Weekday_scheduled"].value_counts()["Vendredi"],data["Weekday_scheduled"].value_counts()["Samedi"],0])

plt.ylabel('Frequency')

plt.title('Weekday_scheduled')

plt.show()
# Frequencies for No-show by Weekday_scheduled

pd.crosstab(data["Weekday_scheduled"],data["No-show"])
# Representation

p1 = plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"], [pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Lundi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Mardi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Mercredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Jeudi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Vendredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Samedi"],0])

p2 = plt.bar(["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"], [pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Lundi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Mardi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Mercredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Jeudi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Vendredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["Yes"]["Samedi"],0],

             bottom=[pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Lundi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Mardi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Mercredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Jeudi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Vendredi"],pd.crosstab(data["Weekday_scheduled"],data["No-show"])["No"]["Samedi"],0])



plt.ylabel('Frequency')

plt.title('No-show by Weekday_scheduled')

plt.legend((p1[0], p2[0]), ('No', 'Yes'))
# Waiting Boxplot 

fig_age, ax1 = plt.subplots()

ax1.set_title('Waiting')

ax1.boxplot(data["Waiting"])
# Histogram

plt.hist(data["Waiting"], rwidth=0.8, bins=20)
# Waiting Boxplot by No-show

fig_age, ax1 = plt.subplots()

ax1.set_title('No-show by Waiting')

ax1.boxplot([data[data["No-show"]=="Yes"]["Waiting"],data[data["No-show"]=="No"]["Waiting"]])

plt.gca().xaxis.set_ticklabels(['No-show : Yes', 'No-show : No'])
# Frequencies for No-show

data["No-show"].value_counts()
# Representation

plt.bar(["No","Yes"],data["No-show"].value_counts())

plt.ylabel('Frequency')

plt.title('No-show')

plt.show()
# Fixing seed

random.seed(1243)



# Encoding

lenc = LabelEncoder()

data.Gender = lenc.fit_transform(data.Gender)

data.Neighbourhood = lenc.fit_transform(data.Neighbourhood)

data.Weekday_Appointment = lenc.fit_transform(data.Weekday_Appointment)

data["No-show"] = lenc.fit_transform(data["No-show"])
# Separating explicative features from explicated one

X = pd.DataFrame(data, columns = ["Gender","Neighbourhood","Scholarship","Hypertension","Diabetes","Alcoholism","Handicap","Waiting","Weekday_Appointment"])

y = data["No-show"]



# Separating data set into train and test

XTrain, XTest, yTrain, yTest = train_test_split(X,y,test_size = 0.25)
# Random Forest

RF = RandomForestClassifier()

RF.fit(XTrain,yTrain)

RF_Prediction = RF.predict(XTest)



# Results

RF_conf = confusion_matrix(yTest,RF_Prediction)

RF_accuracy = (RF_conf[0,0]+RF_conf[1,1])/len(yTest)

RF_sensitivity = RF_conf[0,0]/(RF_conf[0,0]+RF_conf[0,1])

RF_specificity = RF_conf[1,1]/(RF_conf[1,1]+RF_conf[1,0])

print("Accuracy: {:.3f}      Sensitivity: {:.3f}      Specificity: {:.3f}".format(RF_accuracy,RF_sensitivity,RF_specificity))

RF_conf
# LogisticRegression

LR = LogisticRegression()

LR.fit(XTrain,yTrain)

LR_Prediction = LR.predict(XTest)



# Results

LR_conf = confusion_matrix(yTest,LR_Prediction)

LR_accuracy = (LR_conf[0,0]+LR_conf[1,1])/len(yTest)

LR_sensitivity = LR_conf[0,0]/(LR_conf[0,0]+LR_conf[0,1])

LR_specificity = LR_conf[1,1]/(LR_conf[1,1]+LR_conf[1,0])

print("Accuracy: {:.3f}      Sensitivity: {:.3f}      Specificity: {:.3f}".format(LR_accuracy,LR_sensitivity,LR_specificity))

LR_conf
# KNeighbors

KN=KNeighborsClassifier(n_neighbors=3) #Better for 1 or 3 neighbors

KN.fit(XTrain,yTrain)

KN.Prediction=KN.predict(XTest)



# Results

KN_conf=confusion_matrix(yTest,KN.Prediction)

KN_accuracy=(KN_conf[0,0]+KN_conf[1,1])/len(yTest)

KN_sensitivity = KN_conf[0,0]/(KN_conf[0,0]+KN_conf[0,1])

KN_specificity = KN_conf[1,1]/(KN_conf[1,1]+KN_conf[1,0])

print("Accuracy: {:.3f}      Sensitivity: {:.3f}      Specificity: {:.3f}".format(KN_accuracy,KN_sensitivity,KN_specificity))

KN_conf