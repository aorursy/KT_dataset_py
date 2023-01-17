# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, plot_roc_curve, accuracy_score



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



train_folder = "/kaggle/input/healthcare-analytics/Train/"
# take a glance at data

train = pd.read_csv(train_folder+"Train.csv")

train.head()

print(train.describe())

print(train.head())
# plot

train.hist()

plt.tight_layout()
# take a glance at data

test = pd.read_csv(train_folder+"test.csv")

print(test.describe())

print(test.head())
# plot

test.hist()

plt.tight_layout()
# look if duplicate patient_id health_camp_ID combos



# read in the datasets

health_camp_detail = pd.read_csv(train_folder+"Health_Camp_Detail.csv") # join on camp_ID

patient_profile = pd.read_csv(train_folder+"Patient_Profile.csv")# join on Patient_ID

# Health Outcomes

first_healthcamp = pd.read_csv(train_folder+"First_Health_Camp_Attended.csv")

second_healthcamp = pd.read_csv(train_folder+"Second_Health_Camp_Attended.csv")

third_healthcamp = pd.read_csv(train_folder+"Third_Health_Camp_Attended.csv")

print("first_healthcamp")

print(first_healthcamp.head())

print(first_healthcamp.describe())
print("second_healthcamp")

print(second_healthcamp.head())

print(second_healthcamp.describe())
print("third_healthcamp")

print(third_healthcamp.head())

print(third_healthcamp.describe())
# rename column to match other datasets

second_healthcamp.rename(columns={'Health Score': 'Health_Score'}, inplace=True)



# Create a healthscore based on if the patient visited a stall or not

third_healthcamp[['Health_Score']] = third_healthcamp[['Number_of_stall_visited']].where(third_healthcamp[['Number_of_stall_visited']] <= 1, 1) 

# Pick only columns we want to use in the first model

columns = ['Patient_ID',  'Health_Camp_ID',  'Health_Score']

health_scores = pd.concat([first_healthcamp[columns],

                           second_healthcamp[columns],

                           third_healthcamp[columns]])

health_scores
health_scores[['Health_Score']].hist()

plt.tight_layout()
patient_profile = pd.read_csv(train_folder+"Patient_Profile.csv")



print("patient_profile")

print(patient_profile.head())

print(patient_profile.describe())



#plot

patient_profile.hist()

plt.tight_layout()
patient_profile['online_activity'] = patient_profile[['Online_Follower' , 'LinkedIn_Shared',  'Twitter_Shared', 'Facebook_Shared']].max(axis=1)



print(patient_profile[['online_activity']].describe())
# now just remove the columns

patient_profile2 = patient_profile.drop(['Online_Follower' , 'LinkedIn_Shared',  'Twitter_Shared', 'Facebook_Shared'], axis=1)
health_camp_detail = pd.read_csv(train_folder+"Health_Camp_Detail.csv")

print("health_camp_detail")

print(health_camp_detail.tail())

print(health_camp_detail.describe())
# health outcomes

train2 = pd.merge(train, health_scores,  how='inner', left_on=['Patient_ID',  'Health_Camp_ID'], right_on = ['Patient_ID',  'Health_Camp_ID'])

#test2 = pd.merge(test, health_scores,  how='inner', left_on=['Patient_ID',  'Health_Camp_ID'], right_on = ['Patient_ID',  'Health_Camp_ID'])

test2 = test



# patient

train2 = pd.merge(train2, patient_profile2,  how='inner', on=['Patient_ID'])

test2 = pd.merge(test2, patient_profile2,  how='inner', on=['Patient_ID'])



# health camp

train2 = pd.merge(train2, health_camp_detail,  how='inner', on=['Health_Camp_ID'])

test2 = pd.merge(test2, health_camp_detail,  how='inner', on=['Health_Camp_ID'])



y_train = train2[['Health_Score']].round()

X_train = train2.drop(['Patient_ID', 'Health_Camp_ID', 'Health_Score'], axis=1)



# Use this to predict health results

test_final = test2.drop(['Patient_ID', 'Health_Camp_ID'], axis=1)



# replace "none" with nan

X_train[['Income','Education_Score','Age']] = X_train[['Income','Education_Score','Age']].replace("None", np.nan)

test_final[['Income','Education_Score','Age']] = X_train[['Income','Education_Score','Age']].replace("None", np.nan)



# get rid of dates, come back to this later and extract some information out of them

datecols = ['Registration_Date','First_Interaction', 'Camp_Start_Date','Camp_End_Date']

X_train = X_train.drop(datecols, axis=1)

test_final = test_final.drop(datecols, axis=1)



# Factorise categorical variables

cat_columns = X_train.dtypes.pipe(lambda x: x[x == 'object']).index



for c in cat_columns:

    X_train[c] = pd.factorize(X_train[c])[0]

    test_final[c] = pd.factorize(test_final[c])[0]



# print what the datasets look like

print("Columns in Train set")

print(X_train.columns)

print("\nColumns in Test set")

print(test_final.columns)

print(f"\n\nTrain X dimensions: {X_train.shape}")

print(f"Train y dimensions: {y_train.shape}")

print(f"Test X dimensions: {test_final.shape}")

print("\n\nRoughly 2/3 of y values represent positive health outcomes")

print(y_train.astype(int).sum() / len(y_train))

print(y_test.astype(int).sum() / len(y_test))
X_train_NB = X_train

y_train_NB = y_train

X_train, X_test, y_train, y_test = train_test_split(X_train_NB, y_train_NB, test_size=0.1, random_state=0)

gnb = GaussianNB()

gnb.fit(X_train, y_train.values.ravel())

# predict values

y_hat = gnb.predict(X_test)

# accuracy:

print(f"Accuracy is: {sum(y_hat == y_test.values.ravel()) / len(y_hat)}, which is not great, since 67% of health outcomes are positive anyway")

plot_roc_curve(gnb, X_test, y_test)

plt.show()
# pick probability for positive outcome

y_hat_submission = gnb.predict_proba(test_final)[:,1]

y_hat_submission

#create a submission dataset

submission = test[['Patient_ID', 'Health_Camp_ID']]

submission["Outcome"] = y_hat_submission

submission["Outcome"].hist()
# save submission dataset

submission.to_csv('submission.csv', index=False)