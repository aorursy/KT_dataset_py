#Import basic packages

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

#Output plots in notebook

%matplotlib inline 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#Read data

data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
#Check if missing values

data.info()

#luckily, no missing values in this dataset
#Most packages are not able to deal with string variables,

#Therefore we need to convert string to numeric



#First check what string values are

data['sales'].unique(),data['salary'].unique()
#Convert 'sales' and 'salary' to numeric

data['sales'].replace(['sales', 'accounting', 'hr', 'technical', 'support', 'management',

        'IT', 'product_mng', 'marketing', 'RandD'], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace = True)

data['salary'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace = True)
#Correlation Matrix

corr = data.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')

corr
#Extract 'left' column, because 'left' is our target value

corr_left = pd.DataFrame(corr['left'].drop('left'))

corr_left.sort_values(by = 'left', ascending = False)
#Make new features then remove 'number_project'

#The new feature indicate how many hours the employee spend on a single project

data['avg_hour_project'] = (data['average_montly_hours'] * 12) /data['number_project']

data['avg_hour_project_range'] = pd.cut(data['avg_hour_project'], 3)

data[['avg_hour_project_range', 'left']].groupby(['avg_hour_project_range']).mean()
data.loc[data['avg_hour_project'] <= 749.333, 'avg_hour_project'] = 0

data.loc[(data['avg_hour_project'] > 749.333) & (data['avg_hour_project'] <= 1304.667), 'avg_hour_project'] = 1

data.loc[(data['avg_hour_project'] > 1304.667) & (data['avg_hour_project'] <= 1860.00), 'avg_hour_project'] = 2

data.drop(['avg_hour_project_range'], axis = 1, inplace = True)
#Boxplot

#To briefly see the data distribution and if there're outliers

g = sns.FacetGrid(data, col = 'left')

g.map(sns.boxplot, 'time_spend_company')
#Outliers do exist, just drop those observations

dropdata = data[data['time_spend_company'] >= 8]

data.drop(dropdata.index, inplace = True)
#time_spend_company

sns.barplot(x = 'time_spend_company', y = 'left', data = data)

sns.plt.title('Left over time spend at company (barplot)')

sns.factorplot(x= 'time_spend_company', y = 'left', data = data, size = 5)

sns.plt.title('Left over time spend at company (factorplot)')
left = data[data['left'] == 1]

not_left = data[data['left'] == 0]

f, axrrr = plt.subplots(1, 2, sharey=True, sharex = True)



axrrr[0].hist('time_spend_company', data = left, bins = 10)

axrrr[0].set_title('Left')

axrrr[0].set_xlabel('Time Spend at the Company')

axrrr[0].set_ylabel('Number of Observations')

axrrr[1].hist('time_spend_company', data = not_left, bins = 10)

axrrr[1].set_title('Not Left')

axrrr[1].set_xlabel('time_spend_company')

axrrr[1].set_ylabel('Number of Observations')
#time spend with promotion

sns.barplot(x='time_spend_company', y = 'left', hue = 'promotion_last_5years', data = data)
#time spend with work accident

sns.barplot(x='time_spend_company', y = 'left', hue = 'salary', data = data)
#average_monthly_hours

g = sns.FacetGrid(data, hue="left",aspect=4)

g.map(sns.kdeplot,'average_montly_hours',shade= True)

g.set(xlim=(0, data['average_montly_hours'].max()))

g.add_legend()
#Boxplot

g = sns.FacetGrid(data, col = 'left')

g.map(sns.boxplot, 'average_montly_hours')

np.mean(data[data['left']==1]['average_montly_hours']),np.mean(data[data['left']==0]['average_montly_hours'])
#Continuous to categorical

#First create range data using pandas

data['avg_mon_hours_range'] = pd.cut(data['average_montly_hours'], 3)

data[['avg_mon_hours_range', 'left']].groupby(['avg_mon_hours_range']).mean()
#Replace continuous values by categorical ones

data.loc[data['average_montly_hours'] <= 167.333, 'average_montly_hours'] = 0

data.loc[(data['average_montly_hours'] > 167.333) & (data['average_montly_hours'] <= 238.667), 'average_montly_hours'] = 1

data.loc[(data['average_montly_hours'] > 238.667) & (data['average_montly_hours'] <= 310.000), 'average_montly_hours'] = 2

data.drop(['avg_mon_hours_range'], axis = 1, inplace = True)
#number_project

sns.barplot(x = 'number_project', y = 'left', data = data)

sns.plt.title('Left over Number of project')
#Box plot

g = sns.FacetGrid(data, col = 'left')

g.map(sns.boxplot, 'number_project')

print('left_median : ',np.median(data[data['left']==1]['number_project']))

print('not_left_median : ',np.median(data[data['left']==0]['number_project']))

#This indicates number of project is not a good estimator
#satisfaction_level

g = sns.FacetGrid(data, hue="left",aspect=4)

g.map(sns.kdeplot,'satisfaction_level',shade= True)

g.set(xlim=(0, data['satisfaction_level'].max()))

g.add_legend()
#Same as above process

data['satisfaction_range'] = pd.cut(data['satisfaction_level'], 3)

data[['satisfaction_range', 'left']].groupby(['satisfaction_range']).mean()
#last_evaluation

g = sns.FacetGrid(data, hue="left",aspect=4)

g.map(sns.kdeplot,'last_evaluation',shade= True)

g.set(xlim=(0, data['last_evaluation'].max()))

g.add_legend()
data.loc[(data['satisfaction_level'] > 0.697) & (data['satisfaction_level'] <= 1.000), 'satisfaction_level'] = 2

data.loc[(data['satisfaction_level'] > 0.393) & (data['satisfaction_level'] <= 0.697), 'satisfaction_level'] = 1

data.loc[data['satisfaction_level'] <= 0.393, 'satisfaction_level'] = 0



data.drop(['satisfaction_range'], axis = 1, inplace = True)

data.head()
#Same as above process

data['evaluation_range'] = pd.cut(data['last_evaluation'], 3)

data[['evaluation_range','left']].groupby(['evaluation_range']).mean()
data.loc[(data['last_evaluation'] > 0.787) & (data['last_evaluation'] <= 1), 'last_evaluation'] = 2

data.loc[(data['last_evaluation'] > 0.573) & (data['last_evaluation'] <= 0.787), 'last_evaluation'] = 1

data.loc[data['last_evaluation'] <= 0.573, 'last_evaluation'] = 0

data.drop(['evaluation_range'], axis = 1, inplace = True)



data.head()
#salary

sns.barplot('salary', 'left', data = data)

sns.plt.title('Left over Salary (bar plot)')

sns.factorplot('salary','left', data = data, size = 5)

sns.plt.title('Left over Salary (factor plot)')
#we can combine promotion_last_5years to salary to see if what happens

promoted = data[data['promotion_last_5years'] == 1]

not_promoted = data[data['promotion_last_5years'] == 0]
#promotion_last_5years

sns.barplot('promotion_last_5years', 'left', data = data)

sns.plt.title('Left over promotion_last_5years (barplot)')

sns.factorplot('promotion_last_5years','left',order=[0, 1], data=data,size=5)

sns.plt.title('Left over promotion_last_5years (factorplot)')

#it seems people who are promoted in last 5 years are less likely to leave than those who are not.

#Therefore we can confidently say, if someone get promoted, he is much less likely to leave.
#separate employee into promoted and not_promoted groups



fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))



sns.barplot('salary', 'left', data = promoted, ax=axis1)



sns.barplot('salary', 'left', data = not_promoted, ax=axis2)



axis1.set_title('Promoted')



axis2.set_title('Not Promoted')
#Sales

sns.barplot('sales','left',order=[0, 1, 2, 3, 4, 5, 6], data=data)

sns.plt.title('Left over Sales')
#Let's look at our dataset again

data.head()
#Train-Test split

from sklearn.model_selection import train_test_split

label = data.pop('left')

data_train, data_test, label_train, label_test = train_test_split(data, label, test_size = 0.2, random_state = 42)
#Logistic Regression

from sklearn.linear_model import LogisticRegression

logis = LogisticRegression()

logis.fit(data_train, label_train)

logis_score_train = logis.score(data_train, label_train)

print("Training score: ",logis_score_train)

logis_score_test = logis.score(data_test, label_test)

print("Testing score: ",logis_score_test)
#logis.coef_ could help us to see the correlation between features and target value,

#This will not generate correlation values like those in correlation matrix.

#You can treat this as another set of correlation factors

coeff_df = pd.DataFrame(data.columns.delete(0))

coeff_df.columns = ['Features']

coeff_df["Correlation"] = pd.Series(logis.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
#SVM

from sklearn.svm import SVC

svm = SVC()

svm.fit(data_train, label_train)

svm_score_train = svm.score(data_train, label_train)

print("Training score: ",svm_score_train)

svm_score_test = svm.score(data_test, label_test)

print("Testing score: ",svm_score_test)
#kNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(data_train, label_train)

knn_score_train = knn.score(data_train, label_train)

print("Training score: ",knn_score_train)

knn_score_test = knn.score(data_test, label_test)

print("Testing score: ",knn_score_test)
#decision tree

from sklearn import tree

dt = tree.DecisionTreeClassifier()

dt.fit(data_train, label_train)

dt_score_train = dt.score(data_train, label_train)

print("Training score: ",dt_score_train)

dt_score_test = dt.score(data_test, label_test)

print("Testing score: ",dt_score_test)
#random forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(data_train, label_train)

rfc_score_train = rfc.score(data_train, label_train)

print("Training score: ",rfc_score_train)

rfc_score_test = rfc.score(data_test, label_test)

print("Testing score: ",rfc_score_test)
#Model comparison

models = pd.DataFrame({

        'Model'          : ['Logistic Regression', 'SVM', 'kNN', 'Decision Tree', 'Random Forest'],

        'Training_Score' : [logis_score_train, svm_score_train, knn_score_train, dt_score_train, rfc_score_train],

        'Testing_Score'  : [logis_score_test, svm_score_test, knn_score_test, dt_score_test, rfc_score_test]

    })

models.sort_values(by='Testing_Score', ascending=False)