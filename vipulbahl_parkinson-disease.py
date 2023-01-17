import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

import math as math

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer

from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from os import system

from IPython.display import Image

from sklearn.tree import plot_tree

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import StackingClassifier

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn import preprocessing

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from vecstack import stacking

%matplotlib inline

sns.set(color_codes=True)
#step 1.1: Read the dataset

#data=pd.read_csv('Data - Parkinsons.csv')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
PD_file_path='../input/parkinson-disease-identification/Data - Parkinsons.csv'

data=pd.read_csv(PD_file_path)



# step 2.1: browse through the first few columns

data.head()
# Step 2.2: Understand the shape of the data

shape_data=data.shape

print('The shape of the dataframe is',shape_data,'which means there are',shape_data[0],'rows of voice recordings and',shape_data[1],'attributes of patients.')
# Step 2.3: Identify Duplicate records in the data 

# It is very important to check and remove data duplicates. 

# Else our model may break or report overly optimistic / pessimistic performance results

dupes=data.duplicated()

print(' The number of duplicates in the dataset are:',sum(dupes),'\n','Hence, it is quite evident that there are no duplicates in the dataset')
# Step 2.4: Lets analyze the data types

data.info()
# Step 2.5: lets evaluate statistical details of the dataset. 

#We will also add skewness column to the details to get a holistic view of the dataset from statistical perspective

cname=data.columns

data_desc=data.describe().T

data_desc['Skewness']=round(data[cname].skew(),4)

pd.DataFrame(data_desc)
# Step 2.6: Lets visually understand if there is any correlation between the independent variables. 

usecols =[i for i in data.columns if i != ['name','status']]

sns.pairplot(data[usecols]);
# Step 2.7: lets evaluate correlation between different attributes.

# The column name and status has been ignored from the correlation heatmap. 

#The reason for the same will be explained in the next section.

corr=data[usecols].corr()

fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(corr,annot=True,linewidth=0.05,ax=ax, fmt= '.2f');
udata=data.drop('name',axis=1)

udata.head()
# Attributes in the Group

Atr1g2='MDVP:Fo(Hz)'

Atr2g2='MDVP:Fhi(Hz)'

Atr3g2='MDVP:Flo(Hz)'
#EDA 1: 5 point summary to understand spread

Atr1g2_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g2]]

Atr2g2_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g2]]

Atr3g2_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr3g2]]



summ_g2 = pd.concat([Atr1g2_5pt,Atr2g2_5pt,Atr3g2_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 2 are:','\n','\n',summ_g2)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,3,figsize=(16,10)) 

sns.boxplot(x=Atr1g2,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g2,data=udata, ax=ax[1],orient='v')

sns.boxplot(x=Atr3g2,data=udata,ax=ax[2],orient='v')
#EDA 3: Skewness check

Atr1g2_skew=round(stats.skew(udata[Atr1g2]),4)

Atr2g2_skew=round(stats.skew(udata[Atr2g2]),4)

Atr3g2_skew=round(stats.skew(udata[Atr3g2]),4)



print(' The skewness of',Atr1g2,'is', Atr1g2_skew)

print(' The skewness of',Atr2g2,'is', Atr2g2_skew)

print(' The skewness of',Atr3g2,'is', Atr3g2_skew)
##EDA 4: Spread

fig, ax = plt.subplots(1,3,figsize=(16,8)) 

sns.distplot(udata[Atr1g2],ax=ax[0]) 

sns.distplot(udata[Atr2g2],ax=ax[1]) 

sns.distplot(udata[Atr3g2],ax=ax[2])
##EDA 5: Correlation of attributes of group 2 with other attributes.

corr_atr1g2=udata[udata.columns].corr()[Atr1g2][:]

corr_atr2g2=udata[udata.columns].corr()[Atr2g2][:]

corr_atr3g2=udata[udata.columns].corr()[Atr3g2][:]

pd.concat([round(corr_atr1g2,4),round(corr_atr2g2,4),round(corr_atr3g2,4)],axis=1,sort=False).T



# pd.DataFrame(round(corr,4)).T
# Attributes in the Group

Atr1g3='MDVP:Jitter(%)'

Atr2g3='MDVP:Jitter(Abs)'

Atr3g3='MDVP:RAP'

Atr4g3='MDVP:PPQ'

Atr5g3='Jitter:DDP'
#EDA 1: 5 point summary to understand spread

Atr1g3_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g3]]

Atr2g3_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g3]]

Atr3g3_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr3g3]]

Atr4g3_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr4g3]]

Atr5g3_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr5g3]]



summ_g3 = pd.concat([Atr1g3_5pt,Atr2g3_5pt,Atr3g3_5pt,Atr4g3_5pt,Atr5g3_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 3 are:','\n','\n',summ_g3)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,5,figsize=(16,10)) 

sns.boxplot(x=Atr1g3,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g3,data=udata, ax=ax[1],orient='v')

sns.boxplot(x=Atr3g3,data=udata,ax=ax[2],orient='v')

sns.boxplot(x=Atr4g3,data=udata,ax=ax[3],orient='v')

sns.boxplot(x=Atr5g3,data=udata,ax=ax[4],orient='v')
#EDA 3: Skewness check

Atr1g3_skew=round(stats.skew(udata[Atr1g3]),4)

Atr2g3_skew=round(stats.skew(udata[Atr2g3]),4)

Atr3g3_skew=round(stats.skew(udata[Atr3g3]),4)

Atr4g3_skew=round(stats.skew(udata[Atr4g3]),4)

Atr5g3_skew=round(stats.skew(udata[Atr5g3]),4)



print(' The skewness of',Atr1g3,'is', Atr1g3_skew)

print(' The skewness of',Atr2g3,'is', Atr2g3_skew)

print(' The skewness of',Atr3g3,'is', Atr3g3_skew)

print(' The skewness of',Atr4g3,'is', Atr4g3_skew)

print(' The skewness of',Atr5g3,'is', Atr5g3_skew)
##EDA 4: Spread

fig, ax = plt.subplots(1,5,figsize=(16,8)) 

sns.distplot(udata[Atr1g3],ax=ax[0]) 

sns.distplot(udata[Atr2g3],ax=ax[1]) 

sns.distplot(udata[Atr3g3],ax=ax[2])

sns.distplot(udata[Atr4g3],ax=ax[3])

sns.distplot(udata[Atr5g3],ax=ax[4])
##EDA 5: Correlation of attributes of group 3 with other attributes.

corr_atr1g3=udata[udata.columns].corr()[Atr1g3][:]

corr_atr2g3=udata[udata.columns].corr()[Atr2g3][:]

corr_atr3g3=udata[udata.columns].corr()[Atr3g3][:]

corr_atr4g3=udata[udata.columns].corr()[Atr4g3][:]

corr_atr5g3=udata[udata.columns].corr()[Atr5g3][:]

pd.concat([round(corr_atr1g3,4),round(corr_atr2g3,4),round(corr_atr3g3,4),round(corr_atr4g3,4),round(corr_atr5g3,4)],axis=1,sort=False).T
# Attributes in the Group

Atr1g4='MDVP:Shimmer'

Atr2g4='MDVP:Shimmer(dB)'

Atr3g4='Shimmer:APQ3'

Atr4g4='Shimmer:APQ5'

Atr5g4='MDVP:APQ'

Atr6g4='Shimmer:DDA'
#EDA 1: 5 point summary to understand spread

Atr1g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g4]]

Atr2g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g4]]

Atr3g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr3g4]]

Atr4g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr4g4]]

Atr5g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr5g4]]

Atr6g4_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr6g4]]



summ_g4 = pd.concat([Atr1g4_5pt,Atr2g4_5pt,Atr3g4_5pt,Atr4g4_5pt,Atr5g4_5pt,Atr6g4_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 4 are:','\n','\n',summ_g4)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,6,figsize=(16,10)) 

sns.boxplot(x=Atr1g4,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g4,data=udata, ax=ax[1],orient='v')

sns.boxplot(x=Atr3g4,data=udata,ax=ax[2],orient='v')

sns.boxplot(x=Atr4g4,data=udata,ax=ax[3],orient='v')

sns.boxplot(x=Atr5g4,data=udata,ax=ax[4],orient='v')

sns.boxplot(x=Atr6g4,data=udata,ax=ax[5],orient='v')
#EDA 3: Skewness check

Atr1g4_skew=round(stats.skew(udata[Atr1g4]),4)

Atr2g4_skew=round(stats.skew(udata[Atr2g4]),4)

Atr3g4_skew=round(stats.skew(udata[Atr3g4]),4)

Atr4g4_skew=round(stats.skew(udata[Atr4g4]),4)

Atr5g4_skew=round(stats.skew(udata[Atr5g4]),4)

Atr6g4_skew=round(stats.skew(udata[Atr6g4]),4)



print(' The skewness of',Atr1g4,'is', Atr1g4_skew)

print(' The skewness of',Atr2g4,'is', Atr2g4_skew)

print(' The skewness of',Atr3g4,'is', Atr3g4_skew)

print(' The skewness of',Atr4g4,'is', Atr4g4_skew)

print(' The skewness of',Atr5g4,'is', Atr5g4_skew)

print(' The skewness of',Atr6g4,'is', Atr6g4_skew)
##EDA 4: Spread

fig, ax = plt.subplots(1,6,figsize=(16,8)) 

sns.distplot(udata[Atr1g4],ax=ax[0]) 

sns.distplot(udata[Atr2g4],ax=ax[1]) 

sns.distplot(udata[Atr3g4],ax=ax[2])

sns.distplot(udata[Atr4g4],ax=ax[3])

sns.distplot(udata[Atr5g4],ax=ax[4])

sns.distplot(udata[Atr6g4],ax=ax[5])
##EDA 5: Correlation of attributes of group 4 with other attributes.

corr_atr1g4=udata[udata.columns].corr()[Atr1g4][:]

corr_atr2g4=udata[udata.columns].corr()[Atr2g4][:]

corr_atr3g4=udata[udata.columns].corr()[Atr3g4][:]

corr_atr4g4=udata[udata.columns].corr()[Atr4g4][:]

corr_atr5g4=udata[udata.columns].corr()[Atr5g4][:]

corr_atr6g4=udata[udata.columns].corr()[Atr6g4][:]

pd.concat([round(corr_atr1g4,4),round(corr_atr2g4,4),round(corr_atr3g4,4),round(corr_atr4g4,4),round(corr_atr5g4,4),round(corr_atr5g4,4)],axis=1,sort=False).T
# Attributes in the Group

Atr1g5='NHR'

Atr2g5='HNR'
#EDA 1: 5 point summary to understand spread

Atr1g5_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g5]]

Atr2g5_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g5]]



summ_g5 = pd.concat([Atr1g5_5pt,Atr2g5_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 5 are:','\n','\n',summ_g5)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,2,figsize=(16,10)) 

sns.boxplot(x=Atr1g5,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g5,data=udata, ax=ax[1],orient='v')
#EDA 3: Skewness check

Atr1g5_skew=round(stats.skew(udata[Atr1g5]),4)

Atr2g5_skew=round(stats.skew(udata[Atr2g5]),4)



print(' The skewness of',Atr1g5,'is', Atr1g5_skew)

print(' The skewness of',Atr2g5,'is', Atr2g5_skew)

##EDA 4: Spread

fig, ax = plt.subplots(1,2,figsize=(16,8)) 

sns.distplot(udata[Atr1g5],ax=ax[0]) 

sns.distplot(udata[Atr2g5],ax=ax[1]) 
##EDA 5: Correlation of attributes of group 5 with other attributes.

corr_atr1g5=udata[udata.columns].corr()[Atr1g5][:]

corr_atr2g5=udata[udata.columns].corr()[Atr2g5][:]

pd.concat([round(corr_atr1g5,4),round(corr_atr2g5,4)],axis=1,sort=False).T
# Attributes in the Group

Atr1g7='RPDE'

Atr2g7='D2'
#EDA 1: 5 point summary to understand spread

Atr1g7_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g7]]

Atr2g7_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g7]]



summ_g7 = pd.concat([Atr1g7_5pt,Atr2g7_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 7 are:','\n','\n',summ_g7)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,2,figsize=(16,10)) 

sns.boxplot(x=Atr1g7,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g7,data=udata, ax=ax[1],orient='v')
#EDA 3: Skewness check

Atr1g7_skew=round(stats.skew(udata[Atr1g7]),4)

Atr2g7_skew=round(stats.skew(udata[Atr2g7]),4)



print(' The skewness of',Atr1g7,'is', Atr1g7_skew)

print(' The skewness of',Atr2g7,'is', Atr2g7_skew)
##EDA 4: Spread

fig, ax = plt.subplots(1,2,figsize=(16,8)) 

sns.distplot(udata[Atr1g7],ax=ax[0]) 

sns.distplot(udata[Atr2g7],ax=ax[1]) 
##EDA 5: Correlation of attributes of group 7 with other attributes.

corr_atr1g7=udata[udata.columns].corr()[Atr1g7][:]

corr_atr2g7=udata[udata.columns].corr()[Atr2g7][:]

pd.concat([round(corr_atr1g7,4),round(corr_atr2g7,4)],axis=1,sort=False).T
# Attributes in the Group

Atr1g8='DFA'
#EDA 1: 5 point summary to understand spread

Atr1g8_5pt=udata.describe().loc[['min','25%','50%','75%','max','mean'],[Atr1g8]]



print('The 5 point summary of attributes in group 7 are:','\n','\n',Atr1g8_5pt)
#EDA 2: Outliar Detection leveraging Box Plot

sns.boxplot(x=Atr1g8,data=udata,orient='v');
#EDA 3: Skewness check

Atr1g8_skew=round(stats.skew(udata[Atr1g8]),4)



print(' The skewness of',Atr1g8,'is', Atr1g8_skew)
##EDA 4: Spread

sns.distplot(udata[Atr1g8]);
##EDA 5: Correlation of attributes of group 8 with other attributes.

corr_atr1g8=udata[udata.columns].corr()[Atr1g8][:]

pd.DataFrame(round(corr_atr1g8,4)).T
# Attributes in the Group

Atr1g9='spread1'

Atr2g9='spread2'

Atr3g9='PPE'
#EDA 1: 5 point summary to understand spread

Atr1g9_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr1g9]]

Atr2g9_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr2g9]]

Atr3g9_5pt=udata.describe().loc[['min','25%','50%','75%','max'],[Atr3g9]]



summ_g9 = pd.concat([Atr1g9_5pt,Atr2g9_5pt,Atr3g9_5pt],axis=1,sort=False)



print('The 5 point summary of attributes in group 9 are:','\n','\n',summ_g9)
#EDA 2: Outliar Detection leveraging Box Plot

fig, ax = plt.subplots(1,3,figsize=(16,10)) 

sns.boxplot(x=Atr1g9,data=udata, ax=ax[0],orient='v') 

sns.boxplot(x=Atr2g9,data=udata, ax=ax[1],orient='v')

sns.boxplot(x=Atr3g9,data=udata, ax=ax[2],orient='v')
#EDA 3: Skewness check

Atr1g9_skew=round(stats.skew(udata[Atr1g9]),4)

Atr2g9_skew=round(stats.skew(udata[Atr2g9]),4)

Atr3g9_skew=round(stats.skew(udata[Atr3g9]),4)



print(' The skewness of',Atr1g9,'is', Atr1g9_skew)

print(' The skewness of',Atr2g9,'is', Atr2g9_skew)

print(' The skewness of',Atr3g9,'is', Atr3g9_skew)
##EDA 4: Spread

fig, ax = plt.subplots(1,3,figsize=(16,8)) 

sns.distplot(udata[Atr1g9],ax=ax[0]) 

sns.distplot(udata[Atr2g9],ax=ax[1])

sns.distplot(udata[Atr3g9],ax=ax[2])
##EDA 5: Comparison of Three nonlinear measures of fundamental frequency between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,3,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g9,data=udata,ax=ax[0])

sns.boxplot(x='status',y=Atr2g9,data=udata,ax=ax[1])

sns.boxplot(x='status',y=Atr3g9,data=udata,ax=ax[2])
##EDA 6: Correlation of attributes of group 9 with other attributes.

corr_atr1g9=udata[udata.columns].corr()[Atr1g9][:]

corr_atr2g9=udata[udata.columns].corr()[Atr2g9][:]

corr_atr3g9=udata[udata.columns].corr()[Atr3g9][:]

pd.concat([round(corr_atr1g9,4),round(corr_atr2g9,4),round(corr_atr3g9,4)],axis=1,sort=False).T
Atrg6='status'
# EDA 1: Count of subjects who had Parkinson Disease and subjects who did not have Parkinson Disease



udata_yPD= udata[udata[Atrg6]==1]

udata_nPD= udata[udata[Atrg6]==0]

num_yPD=udata[Atrg6][udata[Atrg6]==1].count()

num_nPD=udata[Atrg6][udata[Atrg6]==0].count()

print('The total number of subjects who have Parkinson Disease are',num_yPD,'which is',round(num_yPD/shape_data[0]*100,2),

      'percent of the total dataset.')

print('The total number of subjects who do not have Parkinson Disease are',num_nPD,'which is',round(num_nPD/shape_data[0]*100,2),

      'percent of the total dataset.')
##EDA 2: Vocal frequency comparison between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,3,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g2,data=udata,ax=ax[0])

sns.boxplot(x='status',y=Atr2g2,data=udata,ax=ax[1])

sns.boxplot(x='status',y=Atr3g2,data=udata,ax=ax[2])
##EDA 3: Comparison of measures of variation in fundamental frequency between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,5,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g3,data=udata,ax=ax[0])

sns.boxplot(x='status',y=Atr2g3,data=udata,ax=ax[1])

sns.boxplot(x='status',y=Atr3g3,data=udata,ax=ax[2])

sns.boxplot(x='status',y=Atr4g3,data=udata,ax=ax[3])

sns.boxplot(x='status',y=Atr5g3,data=udata,ax=ax[4])
##EDA 4: Comparison of measures of variation in amplitude between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,6,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g4,data=udata,ax=ax[0],palette="Set1")

sns.boxplot(x='status',y=Atr2g4,data=udata,ax=ax[1],palette="Set1")

sns.boxplot(x='status',y=Atr3g4,data=udata,ax=ax[2],palette="Set1")

sns.boxplot(x='status',y=Atr4g4,data=udata,ax=ax[3],palette="Set1")

sns.boxplot(x='status',y=Atr5g4,data=udata,ax=ax[4],palette="Set1")

sns.boxplot(x='status',y=Atr6g4,data=udata,ax=ax[5],palette="Set1")
##EDA 5: Comparison of Two measures of ratio of noise to tonal components in the voice between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,2,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g5,data=udata,ax=ax[0])

sns.boxplot(x='status',y=Atr2g5,data=udata,ax=ax[1])
##EDA 6: Comparison of Two nonlinear dynamical complexity measures between people having PD (status=1) and people not having PD (status=0).

fig, ax = plt.subplots(1,2,figsize=(16,8))

sns.boxplot(x='status',y=Atr1g7,data=udata,ax=ax[0],palette="Set1")

sns.boxplot(x='status',y=Atr2g7,data=udata,ax=ax[1],palette="Set1")
##EDA 7: Comparison of Signal fractal scaling exponent between people having PD (status=1) and people not having PD (status=0).



sns.distplot( udata[udata.status == 0][Atr1g8], color = 'g')

sns.distplot( udata[udata.status == 1][Atr1g8], color = 'r')



# sns.boxplot(x='status',y=Atr1g8,data=udata,palette="Set1")
# lets make a copy of the data

pdata=udata.copy()


def calc_vif(X):

    vif = pd.DataFrame()

    vif["variables"] = X.columns

    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
calc_vif(round(pdata,2))
# The attribute with the highest VIF is DFA. Dropping DFA from the dataset. 

#We also noticed that VIF of MDVP:Jitter(Abs) is Not a number. Hence, we will drop MDVP:Jitter(Abs) as well.

# we will also drop Status; since it is a target variable

pdata=pdata.drop(['DFA','MDVP:Jitter(Abs)','status'],axis=1)
# computing VIF of remaining attributes

calc_vif(round(pdata,2))
# It is clearly noticable that VIF of other variable has decreased when we dropped DFA from the data set.

# We will continue to delete one attribute at a time and check till the VIF of the remaining attributes is below 10; 

# and we will select the attribute with highest VIF for deletion.

pdata=pdata.drop(['MDVP:Shimmer(dB)','spread1','MDVP:Shimmer','D2','Shimmer:DDA','RPDE','Shimmer:APQ5','MDVP:Fo(Hz)',

                  'PPE','HNR','Shimmer:APQ3','Jitter:DDP'],axis=1)
# computing VIF of remaining attributes

calc_vif(round(pdata,2))
# lets build our classification model

# independent variables

X = pdata

# X=pd.DataFrame(X_stand1)

# the dependent variable

y = udata['status']
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.30,random_state=1)
# lets check split of data

print("{0:0.2f}% data is in training set".format((len(X_train)/len(pdata.index)) * 100))

print("{0:0.2f}% data is in test set".format((len(X_test)/len(pdata.index)) * 100))
y_train_yPD=y_train[y_train==1].count()

y_train_nPD=y_train[y_train==0].count()

y_test_yPD=y_test[y_test==1].count()

y_test_nPD=y_test[y_test==0].count()



print("In the original dataset people who had parkinson Dieases    : {0} ({1:0.2f}%)".format(len(pdata.loc[udata['status'] == 1]), (len(pdata.loc[udata['status'] == 1])/len(pdata.index)) * 100))

print("In the original dataset people who didnot have Parkinson Disease   : {0} ({1:0.2f}%)".format(len(pdata.loc[udata['status'] == 0]), (len(pdata.loc[udata['status'] == 0])/len(pdata.index)) * 100))

print("")

print("In the training dataset people who who had parkinson Dieases    : {0} ({1:0.2f}%)".format(y_train_yPD, (y_train_yPD/len(y_train))*100))

print("In the training dataset people who didnot have Parkinson Disease    : {0} ({1:0.2f}%)".format(y_train_nPD, (y_train_nPD/len(y_train))*100))

print("")

print("In the test dataset people who who had parkinson Dieases    : {0} ({1:0.2f}%)".format(y_test_yPD, (y_test_yPD/len(y_test))*100))

print("In the test dataset people who didnot have Parkinson Disease    : {0} ({1:0.2f}%)".format(y_test_nPD, (y_test_nPD/len(y_test))*100))
# lets create a copy of the train and test data for scaling

X_Train_stand = X_train.copy()

X_Test_stand = X_test.copy()
# we will use standard scaler for scaling the data.

scale = StandardScaler().fit(X_Train_stand)
X_train= scale.transform(X_Train_stand)

X_test= scale.transform(X_Test_stand)
#while we have already checked in the section 2.4 leveraging fuction info() that there are no null values. 

# Still lets identify and count the null values



if (pd.DataFrame(X_train).isnull().sum().any()==0):

    print('There are no null values in the training datset')

else:

    print('There are null values in the training datset')



if (pd.DataFrame(X_test).isnull().sum().any()==0):

    print('There are no null values in the test datset')

else:

    print('There are null values in the test datset')
# Fit the model on train data

model = LogisticRegression(solver="liblinear")

model.fit(X_train,y_train)
# predict on the test data

y_predict_LR = model.predict(X_test)

y_predict_LR
coef_df = pd.DataFrame(model.coef_)

coef_df['intercept'] = model.intercept_

coef_df
model_score_LR = model.score(X_test, y_test)

print("Model Accuracy of Logistic Regression is: {0:.4f}".format(model_score_LR))

print()
print("Confusion Matrix - Logistic Regression")

cm=metrics.confusion_matrix(y_test, y_predict_LR, labels=[1, 0])



df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Logistic Regression")

print(metrics.classification_report(y_test, y_predict_LR, labels=[1, 0]))
# Call Nearest Neighbour algorithm and fit the model on train data

NNH = KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )

NNH.fit(X_train, np.ravel(y_train,order='C'))
# For every test data point, predict it's label based on 5 nearest neighbours in this model. 

#The majority class will be assigned to the test data point



y_predict_KNN = NNH.predict(X_test)

model_score_KNN = NNH.score(X_test, y_test)



print("Model Accuracy of KNN is: {0:.4f}".format(model_score_KNN))

print()
print("Confusion Matrix - KNN")

cm=metrics.confusion_matrix(y_test, y_predict_KNN, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - KNN")

print(metrics.classification_report(y_test, y_predict_KNN, labels=[1, 0]))
NB_model = GaussianNB()

NB_model.fit(X_train, y_train)
y_predict_NB = NB_model.predict(X_test)

model_score_NB=metrics.accuracy_score(y_test, y_predict_NB)



print("Model Accuracy of Naive Bayes is: {0:.4f}".format(model_score_NB))

print()
print("Confusion Matrix - Naive Bayes")

cm=metrics.confusion_matrix(y_test, y_predict_NB, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Naive Bayes")

print(metrics.classification_report(y_test, y_predict_NB, labels=[1, 0]))
clf = svm.SVC(gamma=0.025, C=3) 
clf.fit(X_train , y_train)
y_predict_SVM = clf.predict(X_test)
model_score_NB=metrics.accuracy_score(y_test, y_predict_SVM)



print("Model Accuracy of SVM is: {0:.4f}".format(model_score_NB))

print()
print("Confusion Matrix - SVM")

cm=metrics.confusion_matrix(y_test, y_predict_SVM, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - SVM")

print(metrics.classification_report(y_test, y_predict_SVM, labels=[1, 0]))
models = [

    KNeighborsClassifier(n_neighbors=5),

    RandomForestClassifier(random_state=0, n_estimators=100, max_depth=3),

    XGBClassifier(random_state=0, learning_rate=0.1, n_estimators=100, max_depth=3)

]
S_train, S_test = stacking(models,                   

                           X_train, y_train, X_test,   

                           regression=False, mode='oof_pred_bag',metric=accuracy_score, n_folds=5, 

                           stratified=True, shuffle=True, random_state=0, verbose=2)
model = XGBClassifier(random_state=0, learning_rate=0.1, n_estimators=100, max_depth=3)

model = model.fit(S_train, y_train)

y_predict_Stack1 = model.predict(S_test)

print('The accuracy of the meta Classfier 1 is: [%.8f]' % metrics.accuracy_score(y_test, y_predict_Stack1))
print("Confusion Matrix - Meta Classifier 1")

cm=metrics.confusion_matrix(y_test, y_predict_Stack1, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Meta-Classifier 1")

print(metrics.classification_report(y_test, y_predict_Stack1, labels=[1, 0]))
estimators = [

    ('rf', RandomForestClassifier(random_state=0, n_estimators=100, max_depth=3)),

#     ('lr', LogisticRegression(solver="liblinear")),

    ('knn',KNeighborsClassifier(n_neighbors= 5 , weights = 'distance' )),

    ('XGB',XGBClassifier(random_state=0, learning_rate=0.1, n_estimators=100, max_depth=3))

]
clf = StackingClassifier(

    estimators=estimators, final_estimator=XGBClassifier()

)
model = clf.fit(X_train, y_train)

y_predict_Stack2 = model.predict(X_test)

print('The accuracy of the meta classifier 2 is: [%.8f]' % accuracy_score(y_test, y_predict_Stack2))
print("Confusion Matrix - Meta Classifer 2")

cm=metrics.confusion_matrix(y_test, y_predict_Stack2, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Meta-Classifier 2")

print(metrics.classification_report(y_test, y_predict_Stack2, labels=[1, 0]))
model_rf = RandomForestClassifier(n_estimators = 50,random_state=1,max_features=3) 

model_rf = model_rf.fit(X_train, y_train)
y_predict_rf = model_rf.predict(X_test)

print(model_rf.score(X_test, y_test))
print("Confusion Matrix -Random Forest")

cm=metrics.confusion_matrix(y_test, y_predict_rf, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Random Forest")

print(metrics.classification_report(y_test, y_predict_rf, labels=[1, 0]))
bgcl = BaggingClassifier(n_estimators=50,random_state=1)

bgcl = bgcl.fit(X_train, y_train)
y_predict_bag = bgcl.predict(X_test)

print(bgcl.score(X_test , y_test))
print("Confusion Matrix -Bagging Classifier")

cm=metrics.confusion_matrix(y_test, y_predict_bag, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Bagging Classifier")

print(metrics.classification_report(y_test, y_predict_bag, labels=[1, 0]))
AdaBC = AdaBoostClassifier(n_estimators=50, random_state=1)

#abcl = AdaBoostClassifier( n_estimators=50,random_state=1)

AdaBC = AdaBC.fit(X_train, y_train)
y_predict_ada = AdaBC.predict(X_test)

print(AdaBC.score(X_test , y_test))
print("Confusion Matrix -Ada Boost")

cm=metrics.confusion_matrix(y_test, y_predict_ada, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - Ada Boost")

print(metrics.classification_report(y_test, y_predict_ada, labels=[1, 0]))
model = XGBClassifier(random_state=0, learning_rate=0.1, n_estimators=100, max_depth=4)

model = model.fit(X_train, y_train)

y_predict_XGB = model.predict(X_test)

print('The accuracy of the XGB Classifier is: [%.8f]' % accuracy_score(y_test, y_predict_XGB))
print("Confusion Matrix - XGB Classifier")

cm=metrics.confusion_matrix(y_test, y_predict_XGB, labels=[1, 0])

df_cm = pd.DataFrame(cm, index = [i for i in ["1","0"]],

                  columns = [i for i in ["Predict 1","Predict 0"]])

plt.figure(figsize = (7,5))

sns.heatmap(df_cm, annot=True);
print("Classification Report - XGB Classifier")

print(metrics.classification_report(y_test, y_predict_XGB, labels=[1, 0]))
# Model1: Logistic Summary

Accuracy_LR=round(metrics.accuracy_score(y_test, y_predict_LR),2)

Recall_LR=round(metrics.recall_score(y_test, y_predict_LR),2)

Precision_LR=round(metrics.precision_score(y_test, y_predict_LR),2)

F1_LR=round(metrics.f1_score(y_test, y_predict_LR),2)
# Model2: KNN Summary

Accuracy_KNN=round(metrics.accuracy_score(y_test, y_predict_KNN),2)

Recall_KNN=round(metrics.recall_score(y_test, y_predict_KNN),2)

Precision_KNN=round(metrics.precision_score(y_test, y_predict_KNN),2)

F1_KNN=round(metrics.f1_score(y_test, y_predict_KNN),2)
# Model3:Native Bayes Summary

Accuracy_NB=round(metrics.accuracy_score(y_test, y_predict_NB),2)

Recall_NB=round(metrics.recall_score(y_test, y_predict_NB),2)

Precision_NB=round(metrics.precision_score(y_test, y_predict_NB),2)

F1_NB=round(metrics.f1_score(y_test, y_predict_NB),2)
# Model4:SVM Summary

Accuracy_SVM=round(metrics.accuracy_score(y_test, y_predict_SVM),2)

Recall_SVM=round(metrics.recall_score(y_test, y_predict_SVM),2)

Precision_SVM=round(metrics.precision_score(y_test, y_predict_SVM),2)

F1_SVM=round(metrics.f1_score(y_test, y_predict_SVM),2)
# Model5: Meta Classifier 1 Summary

Accuracy_Stack1=round(metrics.accuracy_score(y_test, y_predict_Stack1),2)

Recall_Stack1=round(metrics.recall_score(y_test, y_predict_Stack1),2)

Precision_Stack1=round(metrics.precision_score(y_test, y_predict_Stack1),2)

F1_Stack1=round(metrics.f1_score(y_test, y_predict_Stack1),2)
# Model6: Meta Classifier 2 Summary

Accuracy_Stack2=round(metrics.accuracy_score(y_test, y_predict_Stack2),2)

Recall_Stack2=round(metrics.recall_score(y_test, y_predict_Stack2),2)

Precision_Stack2=round(metrics.precision_score(y_test, y_predict_Stack2),2)

F1_Stack2=round(metrics.f1_score(y_test, y_predict_Stack2),2)
# Model7: Random Forest Summary

Accuracy_rf=round(metrics.accuracy_score(y_test, y_predict_rf),2)

Recall_rf=round(metrics.recall_score(y_test, y_predict_rf),2)

Precision_rf=round(metrics.precision_score(y_test, y_predict_rf),2)

F1_rf=round(metrics.f1_score(y_test, y_predict_rf),2)
# Model8: Bagging Summary

Accuracy_bag=round(metrics.accuracy_score(y_test, y_predict_bag),2)

Recall_bag=round(metrics.recall_score(y_test, y_predict_bag),2)

Precision_bag=round(metrics.precision_score(y_test, y_predict_bag),2)

F1_bag=round(metrics.f1_score(y_test, y_predict_bag),2)
# Model9: Ada Boost Summary

Accuracy_ada=round(metrics.accuracy_score(y_test, y_predict_ada),2)

Recall_ada=round(metrics.recall_score(y_test, y_predict_ada),2)

Precision_ada=round(metrics.precision_score(y_test, y_predict_ada),2)

F1_ada=round(metrics.f1_score(y_test, y_predict_ada),2)
# Model10: XGB Summary

Accuracy_XGB=round(metrics.accuracy_score(y_test, y_predict_XGB),2)

Recall_XGB=round(metrics.recall_score(y_test, y_predict_XGB),2)

Precision_XGB=round(metrics.precision_score(y_test, y_predict_XGB),2)

F1_XGB=round(metrics.f1_score(y_test, y_predict_XGB),2)
summary = {'Accuracy': [Accuracy_LR,Accuracy_KNN,Accuracy_NB,Accuracy_SVM,Accuracy_Stack1,Accuracy_Stack2,Accuracy_rf,Accuracy_bag,Accuracy_ada,

                        Accuracy_XGB],



                    'Recall': [Recall_LR,Recall_KNN,Recall_NB,Recall_SVM,Recall_Stack1,Recall_Stack2,Recall_rf,Recall_bag,Recall_ada,

                        Recall_XGB],



                     'Precision': [Precision_LR,Precision_KNN,Precision_NB,Precision_SVM,Precision_Stack1,Precision_Stack2,Precision_rf,Precision_bag,Precision_ada,

                        Precision_XGB],

                       

                       'F1Score':[F1_LR,F1_KNN,F1_NB,F1_SVM,F1_Stack1,F1_Stack2,F1_rf,F1_bag,F1_ada,

                        F1_XGB]}



models=['Logistic Regression','KNN','Naive Bayes','SVM','Meta Classifier 1','Meta Classifier 2','Random Forest','Bagging','Ada Boosting','XGB']

sum_df = pd.DataFrame(summary,models)
sum_df