# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

busdata=pd.read_csv("../input/Bus_Breakdown_and_Delays.csv")

busdata.head()

#Lets check how many rows we have...

busdata.tail()

#I wanna classify breakdowns from Running Late from this data

#The data as such has no null or NaN values as checked manually already in this case and there is no need to check this further.

#I would like to encode my data since most columns are categoric and running in classifier on encoded data would give me better results

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

le.fit(busdata['Breakdown_or_Running_Late'])

busdata['Breakdown_or_Running_Late']=le.transform(busdata['Breakdown_or_Running_Late'])

le.fit(busdata['Has_Contractor_Notified_Parents'])

busdata['Has_Contractor_Notified_Parents']=le.transform(busdata['Has_Contractor_Notified_Parents'])

le.fit(busdata['Have_You_Alerted_OPT'])

busdata['Have_You_Alerted_OPT']=le.transform(busdata['Have_You_Alerted_OPT'])

le.fit(busdata['Reason'])

busdata['Reason']=le.transform(busdata['Reason'])

le.fit(busdata['Has_Contractor_Notified_Schools'])

busdata['Has_Contractor_Notified_Schools']=le.transform(busdata['Has_Contractor_Notified_Schools'])

le.fit(busdata['School_Age_or_PreK'])

busdata['School_Age_or_PreK']=le.transform(busdata['School_Age_or_PreK'])

le.fit(busdata['Schools_Serviced'])

busdata['Schools_Serviced']=le.transform(busdata['Schools_Serviced'])

busdata.head()

#Dropping columns which are less significant from judgement

busdata_filtered=busdata.drop(['Busbreakdown_ID','Run_Type','Incident_Number'], axis=1)



#We need to clean the how_long_delayed column since it is alpha numeric.

#To do so i am wrting a function

import re

def howlong(df):

    i=0

    for each in df:

        regex = re.compile('[!^a-zA-Z ]')

        #First parameter is the replacement, second parameter is your input string

        each = regex.sub('', each)

        df[i]=each

        i=i+1

    return df



df_temp=pd.DataFrame()

df_temp['How_Long_Delayed#']=(howlong(busdata_filtered['How_Long_Delayed'].head()))



#Has it converted?

df_temp.head()



#Yes, now i would add it to the original dataframe

busdata_filtered['How_Long_Delayed']=df_temp

busdata_filtered.head()

#Looks better now

#Lets drop the "How long delayed parameter for now", and include it later after the first model is decent enough

busdata_clean=busdata_filtered.drop(['How_Long_Delayed','Bus_No','Bus_Company_Name','School_Year','Schools_Serviced','Occurred_On','Created_On','Boro','Informed_On','Last_Updated_On'],axis=1)

#Lets encode the company name as well as there are categories

#le.fit(busdata_clean['Bus_Company_Name'])

#busdata_clean['Bus_Company_Name']=le.transform(busdata_clean['Bus_Company_Name'])

busdata_clean.head()



from sklearn import tree

from sklearn import cross_validation



#With these many factors i would prefer to look at a single decision tree first

#Encoding the Bus Number and Route Number as well due to alpha numeric char and this will have not impact on the final model

#le.fit(busdata_clean['Bus_No'])

#busdata_clean['Bus_No']=le.transform(busdata_clean['Bus_No'])



le.fit(busdata_clean['Route_Number'])

busdata_clean['Route_Number']=le.transform(busdata_clean['Route_Number'])



busdata_clean.head()



#Function for float or int conversion

def int_or_float(s):

    try:

        return int(s)

    except ValueError:

        return float(s)



#Converting the data type for columns to keep them int / float instead of string for tree model

list_column=pd.DataFrame()

def convertDataType(df):

    tempCol=[]

    for k,v in df[0:len(df)].items():

        for eachVal in range(len(df[0:len(df)])):

            if(type(v[eachVal]) != float):

                tempCol.append(int_or_float(v[eachVal]))

        list_column[k]=tempCol

        tempCol=[]

    #return list_column





convertDataType(busdata_clean)

list_column.head()



#Now we have the data cleaned for a model

list_column.columns



#Creating the training and test datasets

train, test = cross_validation.train_test_split(list_column, train_size=0.7, test_size=0.3)



x_train=train.drop(['Breakdown_or_Running_Late'],axis=1)

y_train=train['Breakdown_or_Running_Late']



#Run the decision tree with gini index

dt=tree.DecisionTreeClassifier()

dt.fit(x_train,y_train)

dt_train_model=dt.predict(x_train)

pd.crosstab(dt_train_model,y_train)



#Getting the score directly

dt.score(x_train,y_train)



#testing it on test data

x_test=test.drop(['Breakdown_or_Running_Late'],axis=1)

y_test=test['Breakdown_or_Running_Late']

dt_test_model=dt.predict(x_test)

dt.score(x_test,y_test)



#Thats a pretty decent score for a test run but there is significant variation between the training and test data.

#I wouldn't call it a good model based the above condition

#But nevertheless, a singe decision tree did give a good result



#I would go in for an ensemble algorithm in this case.

from sklearn.ensemble import RandomForestClassifier

#Increasing the estimators, since 10 gave a very similar result as that of 1 decision tree

#A max depth of 8 gives us the same result on training and on test. Any other result shows a deviation from traingin and test.

rf=RandomForestClassifier(n_estimators=70, criterion='gini', max_depth=7)

rf.fit(x_train,y_train)

rf_model=rf.predict(x_train)

pd.crosstab(rf_model,y_train)

rf.score(x_train,y_train)

#Thats a 92.25% accurate



rf_test_model=rf.predict(x_test)

pd.crosstab(rf_test_model,y_test)

rf.score(x_test,y_test)

#Thats a 92.28% accuracy on test data and is in accordance with the test data which is sort of more dependable for a predictive model.

#The next steps would be to add the "How long delayed parameter" after cleaning it and then boost.