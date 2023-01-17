#Importing Libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from numpy import mean

from numpy import std

from scipy.stats import norm

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from scipy.stats import boxcox

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import RFE

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix,recall_score,precision_score,accuracy_score

from sklearn.pipeline import Pipeline

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import cross_val_score

# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
#Loading the data

bank_data=pd.read_csv('../input/bank-term-deposit-subscription/bank-additional-full.csv',sep=';')

    

#Let's have a look at our dataset to observe the summary stats for each variable, number of missing data, etc.

print('Number of nulls in the data',bank_data.isnull().any())
bank_data.info()
#Observing the stats

bank_data.describe()
modified_bank_data=bank_data.copy()
#Checking if the data has duplicate values

def duplicate_rows(df):

    print('Number of duplicate records=',df.duplicated().sum())
duplicate_rows(modified_bank_data)
#Dropping the duplicate values

def drop_duplicates(df):

    df=df.drop_duplicates()
drop_duplicates(modified_bank_data)
#Observing the class instances

classes=modified_bank_data.y.value_counts()/len(modified_bank_data)

classes
#Checking the characteristics of the two classes 0 and 1.

modified_bank_data.groupby('y').mean()
#Observing the stats of the categorical features.

modified_bank_data.describe(include=['O'])
#Observing the distribution

modified_bank_data.hist(figsize=(10,10))

plt.show()
#Checking the extreme values of age feature

modified_bank_data[modified_bank_data['age']<18].sort_values(by=['age'])
modified_bank_data=modified_bank_data[modified_bank_data['age']>18]
# Plotting sorted age against their index

sorted_age = sorted(modified_bank_data['age'])

idx = []

for i in range(len(sorted_age)):

    idx.append(i)

x = idx

y = sorted_age





plt.figure(figsize=(10,10))

plt.scatter(x, y, s=10)

plt.axhline(y=0, linestyle='--', color='r')

plt.axhline(y=100, linestyle='--', color='r')
#Observing categorical plots.

def subplots(size,cols,data):

    plt.figure(figsize=size)

    for i in range(len(features)):

        plt.subplot(4,3,i+1)

        sns.countplot(x=data[features[i]])

        



features= modified_bank_data.select_dtypes(exclude=['int64','float64']).columns

    

subplots((30,15),features,modified_bank_data)
#Observing the distribution and extreme values in numerical features

def box_plots(size,cols,data):

    plt.figure(figsize=size)

    for i in range(len(features)):

        plt.subplot(4,3,i+1)

        sns.boxplot(x=data[features[i]])

        

features=modified_bank_data.select_dtypes(exclude=['object']).columns



box_plots((15,15),features,modified_bank_data)

    
#Observing the distribution and extreme values in numerical features

def dist_plots(size,cols,data):

    plt.figure(figsize=size)

    for i in range(len(features)):

        plt.subplot(4,3,i+1)

        sns.distplot(data[features[i]],fit=norm,kde=False,color='y')

        

features=modified_bank_data.select_dtypes(exclude=['object']).columns



dist_plots((20,20),features,modified_bank_data)
#Correlation matrix



def correlation_matrix(data):

    corr=data.corr()

    plt.figure(figsize=(20,10))

    sns.heatmap(corr,annot=True,fmt='.1g',xticklabels=corr.columns.values,yticklabels=corr.columns.values,cmap="YlGnBu",cbar=False)

    

correlation_matrix(modified_bank_data)

    
#As the data is skewed so we will use IQR to identify outliers

cols=modified_bank_data.select_dtypes(exclude=['object']).columns



def outlier_IQR(df,column):



    stat = df[column].describe()

    print(stat)

    IQR = stat['75%'] - stat['25%']

    upper = stat['75%'] + 1.5 * IQR

    lower = stat['25%'] - 1.5 * IQR

    print('The upper and lower bounds for suspected outliers of',each, 'are {} and {}.'.format(upper, lower))
modified_bank_data.loc[modified_bank_data['campaign']>10,'campaign'].value_counts()[-6:]
#Dropping the extreme value row.

modified_bank_data=modified_bank_data[modified_bank_data['campaign']<56]
#Firstly, we log transform the skewed feature age



modified_bank_data['age']=np.log(modified_bank_data['age'])

sns.distplot(modified_bank_data['age'],fit=norm,kde=False)

#Reciprocal transform campaign

modified_bank_data['campaign']=boxcox(modified_bank_data['campaign'],-1)

sns.distplot(modified_bank_data['campaign'],fit=norm,kde=False)
#Dropping the original duration  feature, duration because it is said to influence our model.

modified_bank_data.drop(['duration'],inplace=True,axis=1)
#Checking if there are any instances where the customer was not contacted previously and no calls were made to the

#customer before this campaign.



modified_bank_data[(modified_bank_data['previous']==0) & (modified_bank_data['pdays']==999)]
#Creating customer contacted or not contacted column

def customer_contacted(modified_bank_data):

    if modified_bank_data['previous']==0 and modified_bank_data['pdays']==999:

        return 'No Contact'

    else:

        return 'Contacted'

    

modified_bank_data['Contact_before_this_campaign']= modified_bank_data.apply(customer_contacted, axis=1)



modified_bank_data.Contact_before_this_campaign.value_counts()

    
#26.5% of contacted people bought the term deposit



print('Percentage of contacted people who bought the term dep.=',len(modified_bank_data[(modified_bank_data['Contact_before_this_campaign']=='Contacted') & (modified_bank_data['y']=='yes')])/len(modified_bank_data[modified_bank_data['Contact_before_this_campaign']=='Contacted'])*100)



# 73.4% of the contacted people did not buy the term deposit



print('Percentage of contacted people who did not purchase the term dep.=',len(modified_bank_data[(modified_bank_data['Contact_before_this_campaign']=='Contacted') & (modified_bank_data['y']=='no')])/len(modified_bank_data[modified_bank_data['Contact_before_this_campaign']=='Contacted'])*100)



#8.83% of the non-contacted people had purchased the term deposit



print('Percent of not contacted people who bougth the term dep.=',len(modified_bank_data[(modified_bank_data['Contact_before_this_campaign']=='No Contact') & (modified_bank_data['y']=='yes')])/len(modified_bank_data[modified_bank_data['Contact_before_this_campaign']=='No Contact'])*100)



#91.17% of the not contacted people did not buy the term deposit, so the bank may have to increase their contacted calls 

#Or the bank knew that these category of the people would not buy the term deposit.



print('Percent of not contacted people who bougth the term dep.=',len(modified_bank_data[(modified_bank_data['Contact_before_this_campaign']=='No Contact') & (modified_bank_data['y']=='no')])/len(modified_bank_data[modified_bank_data['Contact_before_this_campaign']=='No Contact'])*100)
#Observing what percentage of people who have an existing loan or default purchased the term deposit.



print('Percent of people who have an existing money to pay and have not purchased the term deposit=',len(modified_bank_data[((modified_bank_data['default']=='yes')|(modified_bank_data['housing']=='yes')|(modified_bank_data['loan']=='yes'))&(modified_bank_data['y']=='no')])/(len(modified_bank_data))*100)
#Replacing the yes and no with 1 and 0

modified_bank_data['y'].replace(['yes','no'],[1,0],inplace=True)
### One hot encoding the features.

modified_bank_data=pd.concat([pd.get_dummies(data=modified_bank_data,columns=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','Contact_before_this_campaign'])

],axis=1)
from sklearn.model_selection import train_test_split as sklearn_train_test_split



def split(df):

    

    X= df[[i for i in list(df.columns) if i!='y']].values

    y=df['y']

    

    X_train,X_test,y_train,y_test=sklearn_train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)

    

    #Scaling the splitted data

    scaler=StandardScaler()

    scaler.fit(X_train)

    X_train=scaler.transform(X_train)

    X_test=scaler.transform(X_test)

    return X_train,X_test,y_train,y_test

X_train,X_test,y_train,y_test=split(modified_bank_data)
#Summary metrics



def summary_metrics(y_test,y_pred):

    conf_matrix= confusion_matrix(y_test, y_pred)

    

    print('confusion matrix',conf_matrix)

    

    

    print('Accuracy',accuracy_score(y_test,y_pred))

    print('Precision',precision_score(y_test,y_pred))

    print('Recall',recall_score(y_test,y_pred))

    
#Observing the benchmark models



y_preds=[]

classification_reports=[]





#Creating instances for three different models

logistic_model=LogisticRegression(max_iter=7600)

Decisiontree_model=DecisionTreeClassifier()

RF_model=RandomForestClassifier()



list_models=[logistic_model,Decisiontree_model,RF_model]





    #Fitting the models

for each in list_models:

    initial_models=each.fit(X_train,y_train)

    y_pred=initial_models.predict(X_test)

    y_preds.append(y_pred)

    classification_reports.append(classification_report(y_test,y_pred))

    summary_metrics(y_test,y_pred)

#Observing the performance of all the three models.

for each in classification_reports:

    print(each)
# Using SMOTE AND TOMEK Over sampling and undersampling techniques



Xdash= modified_bank_data[[i for i in list(modified_bank_data.columns) if i!='y']]

columns=Xdash.columns



from imblearn.combine import SMOTETomek



smt=SMOTETomek(sampling_strategy= 'auto')

X_smt,y_smt=smt.fit_sample(X_train,y_train)

X_smt_df=pd.DataFrame(data=X_smt,columns=columns)

y_smt_df=pd.DataFrame(data=y_smt,columns=['y'])

print('Number of NO subscription in oversampled data',len(y_smt_df[y_smt_df['y']==0]))

print('Number of YES subscription in oversampled data',len(y_smt_df[y_smt_df['y']==1]))
# Applying RFE for feature selection and applying on three different models.

#Also capturing their classification scores.





def elimination_crossval(model):

    

    #Initiating the RFE instance

    rfe=RFE(estimator=RandomForestClassifier(),n_features_to_select=10)

    

    #Fitting the rfe

    X_rfe=rfe.fit_transform(X_smt,y_smt)

    

    #Transforming X_test

    X_rfe_test=rfe.transform(X_test)

    

    model=model

    

    #Creating pipeling to avoid data leakage

    pipeline=Pipeline(steps=[('s',rfe),('m',model)])

    

    cv=RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)

    

    scores =cross_val_score(pipeline,X_rfe, y_smt, scoring='accuracy', cv=cv, n_jobs=-1)

    

    print('Accuracy for model with cross val: %.3f (%.3f)' % (mean(scores)*100, std(scores)*100))

    

    #Fitting the pipeline

    fitted_model=pipeline.fit(X_rfe,y_smt)

    

    y_preds=fitted_model.predict(X_rfe_test)

    

    #Printing the classification report

    print(classification_report(y_test,y_preds))

    

    summary_metrics(y_test,y_preds)
#Logistic Model

elimination_crossval(LogisticRegression(max_iter=7600))
#Decision tree 

elimination_crossval(DecisionTreeClassifier())
#Random forest

elimination_crossval(RandomForestClassifier())
def important_features(estimator,n_features_to_select):

    

    rfe=RFE(estimator=estimator, n_features_to_select=n_features_to_select)

    X_rfe=rfe.fit_transform(X_smt,y_smt)





    columns = X_smt_df.columns

    val = pd.Series(rfe.support_,index = columns)

    features_chosen_rfe = val[val==True].index 

    print(features_chosen_rfe)
important_features(RandomForestClassifier(),4)
modified_bank_data.to_csv('BANK_DATA',index=False)