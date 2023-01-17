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
filepath= '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'



traindata = pd.read_csv(filepath)



filepaths= '/kaggle/input/house-prices-advanced-regression-techniques/test.csv'



testdata = pd.read_csv(filepaths)

testdatacopy=testdata



traindata.head()
#!/usr/bin/env python

# coding: utf-8



# In[1]:





import pandas as pd

import seaborn as sns

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import PolynomialFeatures

from sklearn.tree import DecisionTreeRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

import math

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier





# In[2]:





#filename=r'C:\Users\Vibeesh\Desktop\kaggle data\HousePrices\train.csv'

#traindata=pd.read_csv(filename)

#traindata.head()





# In[3]:





##EDA

#Here we see that there are a large number of null values in 

sns.heatmap(traindata.isnull(),yticklabels=False,cbar=False,cmap='viridis')





# In[4]:





## Now we deal with the columns one by one.

#The ID of the house does not influence the sales price. So, we remove them completely.

traindata=traindata.drop('Id',axis=1)

traindata.head()





# In[5]:





#The MSSubClass column is does not need to be processed.



#Now, We check the MSZoning column.

traindata['MSZoning'].unique()





# In[6]:





#The MSZoning column contains categorical variables so One Hot Encoding must be done.

one_hot = pd.get_dummies(traindata['MSZoning'])

# Drop column Product as it is now encoded

traindata = traindata.drop('MSZoning',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[7]:





#The LotFrontage column as seen in the graph, contains many null values. We need to replace the null values by the mean of the entire column



traindata['LotFrontage'].fillna((traindata['LotFrontage'].mean()), inplace=True)  

traindata.head()





# In[8]:





#The ALley column as seen in the graph has too many missing values. Therefore we remove it completely.

traindata=traindata.drop('Alley',axis=1)

traindata.head()





# In[9]:





#Now we analyse the utilities column.

traindata['Utilities'].unique()

traindata['Utilities'] = traindata['Utilities'].replace(np.nan, 'Utilities', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['Utilities'] = 'Utilities-' + traindata['Utilities'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Utilities'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Utilities',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()



# In[10]:





#We see that this column contains categorical variables and must be replaced by numbers 1 and 0-



traindata.head()





# In[11]:





#Now we analyse the LotArea column. .... this column does not contain any null values and can be left as it is

traindata['LotArea'].isnull().sum(axis = 0)





# In[12]:





# We analyse the street column now-

traindata['Street'].unique()





# In[13]:





traindata['Street'].isnull().sum(axis = 0)





# In[14]:





#This column contains categorical variables and must be converted to numerical form-

traindata['Street'] = traindata['Street'].replace('Pave', 0)

traindata['Street'] = traindata['Street'].replace('Grvl', 1)

traindata.head()





# In[15]:





# We analyse the LotShape column now-

traindata['LotShape'].unique()





# In[16]:





traindata['LotShape'].isnull().sum(axis = 0)





# In[17]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['LotShape'])

# Drop column Product as it is now encoded

traindata = traindata.drop('LotShape',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[18]:





#Now we need to analyse the LandContour

traindata['LandContour'].unique()





# In[19]:





traindata['LandContour'].isnull().sum(axis = 0)





# In[20]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['LandContour'])

# Drop column Product as it is now encoded

traindata = traindata.drop('LandContour',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[21]:





#Now we need to analyse the LotConfig

traindata['LotConfig'].unique()





# In[22]:





traindata['LotConfig'].isnull().sum(axis = 0)





# In[23]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['LotConfig'])

# Drop column Product as it is now encoded

traindata = traindata.drop('LotConfig',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[24]:





#Now we need to analyse the LandSlope

traindata['LandSlope'].unique()





# In[25]:





traindata['LandSlope'].isnull().sum(axis = 0)





# In[26]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['LandSlope'])

# Drop column Product as it is now encoded

traindata = traindata.drop('LandSlope',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[27]:





#Now we need to analyse the LandSlope

traindata['Neighborhood'].unique()





# In[28]:





traindata['Neighborhood'].isnull().sum(axis = 0)





# In[29]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Neighborhood'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Neighborhood',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[30]:





#Now we need to analyse the Condition1

traindata['Condition1'].unique()





# In[31]:





traindata['Condition1'].isnull().sum(axis = 0)





# In[32]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Condition1'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Condition1',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[33]:





#Now we need to analyse the Condition2

traindata['Condition2'].unique()





# In[34]:





traindata['Condition2'].isnull().sum(axis = 0)





# In[35]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for Condition1

traindata['Condition2'] = 'condition2-' + traindata['Condition2'].astype(str)

traindata.head()





# In[36]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Condition2'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Condition2',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[37]:





#Now we need to analyse the BldgType

traindata['BldgType'].unique()





# In[38]:





traindata['BldgType'].isnull().sum(axis = 0)





# In[39]:







#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BldgType'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BldgType',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[40]:





#Now we need to analyse the HouseStyle

traindata['HouseStyle'].unique()





# In[41]:





traindata['HouseStyle'].isnull().sum(axis = 0)





# In[42]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['HouseStyle'])

# Drop column Product as it is now encoded

traindata = traindata.drop('HouseStyle',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[43]:





# Now we analyse the Overall Quality- There are no null values so they are used as given.

traindata['OverallQual'].isnull().sum(axis = 0)





# In[44]:





# Now we analyse the Overall Quality- There are no null values so they are used as given.

traindata['OverallCond'].isnull().sum(axis = 0)





# In[45]:





#Now we analyse the year built-There are no null values so they are used as given.

traindata['YearBuilt'].isnull().sum(axis = 0)





# In[46]:





#Now we analyse the year remodelled column-There are no null values so they are used as given.

traindata['YearRemodAdd'].isnull().sum(axis = 0)





# In[47]:





#Now we analyse the RoofStyle column

traindata['RoofStyle'].isnull().sum(axis = 0)





# In[48]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['RoofStyle'])

# Drop column Product as it is now encoded

traindata = traindata.drop('RoofStyle',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[49]:





#Now we analyse the RoofMatl column

traindata['RoofMatl'].isnull().sum(axis = 0)





# In[50]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['RoofMatl'])

# Drop column Product as it is now encoded

traindata = traindata.drop('RoofMatl',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[51]:





#Now we analyse the Exterior1st column

traindata['Exterior1st'].isnull().sum(axis = 0)





# In[52]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Exterior1st'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Exterior1st',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[53]:





#Now we analyse the Exterior2nd column

traindata['Exterior2nd'].isnull().sum(axis = 0)





# In[54]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for Condition1

traindata['Exterior2nd'] = 'Exterior2nd-' + traindata['Exterior2nd'].astype(str)

traindata.head()





# In[55]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Exterior2nd'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Exterior2nd',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[56]:





#Now we analyse the Exterior2nd column

traindata['MasVnrType'].isnull().sum(axis = 0)





# In[57]:





#We replace the null values with "None"

traindata['MasVnrType'] = traindata['MasVnrType'].replace(np.nan, 'None', regex=True)





# In[58]:





traindata['MasVnrType'].isnull().sum(axis = 0)





# In[59]:





traindata['MasVnrType'].unique()





# In[60]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for a previous column1

traindata['MasVnrType'] = 'MasVnrType-' + traindata['MasVnrType'].astype(str)

traindata.head()





# In[61]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['MasVnrType'])

# Drop column Product as it is now encoded

traindata = traindata.drop('MasVnrType',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[62]:





#Now we analyse the Exterior2nd column

traindata['MasVnrArea'].isnull().sum(axis = 0)





# In[63]:





#We replace the null values with "None"

traindata['MasVnrArea'] = traindata['MasVnrArea'].replace(np.nan, 0, regex=True)





# In[64]:





traindata['MasVnrArea'].isnull().sum(axis = 0)





# In[65]:





traindata['ExterQual'].isnull().sum(axis = 0)





# In[66]:





traindata['ExterQual'].unique()





# In[67]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['ExterQual'])

# Drop column Product as it is now encoded

traindata = traindata.drop('ExterQual',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[68]:





#Analyse ExterCond

traindata['ExterCond'].isnull().sum(axis = 0)





# In[69]:





#We need to change the names of the variables in ExterCond in order to perform OneHotENcoding so that they do not overlap with the OHE done for ExterQual

traindata['ExterCond'] = 'xterCond-' + traindata['ExterCond'].astype(str)

traindata.head()





# In[70]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['ExterCond'])

# Drop column Product as it is now encoded

traindata = traindata.drop('ExterCond',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[71]:





#We analyse Foundation

traindata['Foundation'].isnull().sum(axis = 0)





# In[72]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

traindata['Foundation'] = 'Foundation-' + traindata['Foundation'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Foundation'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Foundation',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[73]:





#Now we analysse BsmtQual

traindata['BsmtQual'].value_counts(dropna=False)





# In[74]:





traindata['BsmtQual'] = traindata['BsmtQual'].replace(np.nan, 'None1', regex=True)





# In[75]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

traindata['BsmtQual'] = 'BsmtQual-' + traindata['BsmtQual'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BsmtQual'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BsmtQual',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[76]:





#We analyse BsmtExposure

traindata['BsmtExposure'].isnull().sum(axis = 0)





# In[77]:





traindata['BsmtExposure'] = traindata['BsmtExposure'].replace(np.nan, 'No', regex=True)





# In[78]:





traindata['BsmtExposure'].unique()





# In[79]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

traindata['BsmtExposure'] = 'BsmtExposure-' + traindata['BsmtExposure'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BsmtExposure'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BsmtExposure',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[80]:





#We analyse BsmtFinSF1 ....There is no null values so we dont have to edit it.

traindata['BsmtFinSF1'].isnull().sum(axis = 0)





# In[81]:





#We analyse BsmtFinType2

traindata['BsmtFinType2'].isnull().sum(axis = 0)





# In[82]:





traindata['BsmtFinType2'].value_counts(dropna=False)





# In[83]:





traindata['BsmtFinType2'] = traindata['BsmtFinType2'].replace(np.nan, 'Unf', regex=True)





# In[84]:





traindata['BsmtFinType2'].isnull().sum(axis = 0)





# In[85]:





#We need to change the names of the variables in BsmtFinType2 in order to perform OneHotENcoding

traindata['BsmtFinType2'] = 'BsmtFinType2-' + traindata['BsmtFinType2'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BsmtFinType2'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BsmtFinType2',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[86]:





#We analyse BsmtFinType2...There is no null values so we dont have to edit it.

traindata['BsmtFinSF2'].isnull().sum(axis = 0)





# In[87]:





#We analyse BsmtUnfSF...There is no null values so we dont have to edit it.

traindata['BsmtUnfSF'].isnull().sum(axis = 0)





# In[88]:





#We analyse TotalBsmtSF...There is no null values so we dont have to edit it.

traindata['TotalBsmtSF'].isnull().sum(axis = 0)





# In[89]:





#We analyse Heating...There is no null values 

traindata['Heating'].isnull().sum(axis = 0)





# In[90]:





traindata['Heating'].unique()





# In[91]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Heating'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Heating',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[92]:





#We analyse HeatingQC...There is no null values 

traindata['HeatingQC'].isnull().sum(axis = 0)





# In[93]:





#We need to change the names of the variables in HeatingQC in order to perform OneHotENcoding

traindata['HeatingQC'] = 'HeatingQC-' + traindata['HeatingQC'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['HeatingQC'])

# Drop column Product as it is now encoded

traindata = traindata.drop('HeatingQC',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[94]:





#We analyse CentralAir...There is no null values 

traindata['CentralAir'].isnull().sum(axis = 0)





# In[95]:





traindata['CentralAir'].unique()





# In[96]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['CentralAir'])

# Drop column Product as it is now encoded

traindata = traindata.drop('CentralAir',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[97]:





#We analyse Electrical...

traindata['Electrical'].isnull().sum(axis = 0)





# In[98]:





traindata['Electrical'].unique()





# In[99]:





traindata['Electrical'].value_counts()





# In[100]:





traindata['Electrical'] = traindata['Electrical'].replace(np.nan, 'SBrkr ', regex=True)





# In[101]:





traindata['Electrical'].isnull().sum(axis = 0)





# In[102]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Electrical'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Electrical',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[103]:





#We analyse 1stFlrSF... There are no null values so we do not need to edit them

traindata['1stFlrSF'].isnull().sum(axis = 0)





# In[104]:





#We analyse 2ndFlrSF... There are no null values so we do not need to edit them

traindata['2ndFlrSF'].isnull().sum(axis = 0)





# In[105]:





#We analyse LowQualFinSF... There are no null values so we do not need to edit them

traindata['LowQualFinSF'].isnull().sum(axis = 0)





# In[106]:





traindata['LowQualFinSF'].unique()





# In[107]:





#We analyse GrLivArea... There are no null values so we do not need to edit them

traindata['GrLivArea'].isnull().sum(axis = 0)





# In[108]:





#We analyse BsmtFullBath... There are no null values so we do not need to edit them

traindata['BsmtFullBath'].isnull().sum(axis = 0)





# In[109]:





#We analyse BsmtHalfBath... There are no null values so we do not need to edit them

traindata['BsmtHalfBath'].isnull().sum(axis = 0)





# In[110]:





#We analyse FullBath... There are no null values so we do not need to edit them

traindata['FullBath'].isnull().sum(axis = 0)





# In[111]:





#We analyse HalfBath... There are no null values so we do not need to edit them

traindata['HalfBath'].isnull().sum(axis = 0)





# In[112]:





#We analyse BedroomAbvGr... There are no null values so we do not need to edit them

traindata['BedroomAbvGr'].isnull().sum(axis = 0)





# In[113]:





#We analyse KitchenAbvGr... There are no null values so we do not need to edit them

traindata['KitchenAbvGr'].isnull().sum(axis = 0)





# In[114]:





#We analyse KitchenQual... There are no null values

traindata['KitchenQual'].isnull().sum(axis = 0)





# In[115]:





#We need to change the names of the variables in BsmtFinType2 in order to perform OneHotENcoding

traindata['KitchenQual'] = 'KitchenQual-' + traindata['KitchenQual'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['KitchenQual'])

# Drop column Product as it is now encoded

traindata = traindata.drop('KitchenQual',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[116]:





#We analyse TotRmsAbvGrd... There are no null values

traindata['TotRmsAbvGrd'].isnull().sum(axis = 0)





# In[117]:





#We analyse Functional... There are no null values

traindata['Functional'].isnull().sum(axis = 0)





# In[118]:





#We analyse Fireplaces... There are no null values

traindata['Fireplaces'].isnull().sum(axis = 0)





# In[119]:





#We analyse FireplaceQu... 

traindata['FireplaceQu'].isnull().sum(axis = 0)





# In[120]:





traindata['FireplaceQu'] = traindata['FireplaceQu'].replace(np.nan, 'NoFireplace', regex=True)





# In[121]:





traindata['FireplaceQu'].isnull().sum(axis = 0)





# In[122]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['FireplaceQu'] = 'FireplaceQu-' + traindata['FireplaceQu'].astype(str)

traindata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['FireplaceQu'])

# Drop column Product as it is now encoded

traindata = traindata.drop('FireplaceQu',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[123]:





#We analyse GarageType... There are no null values

traindata['GarageType'].isnull().sum(axis = 0)





# In[124]:





traindata['GarageType'] = traindata['GarageType'].replace(np.nan, 'Nocar', regex=True)





# In[125]:







#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['GarageType'] = 'GarageType-' + traindata['GarageType'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['GarageType'])

# Drop column Product as it is now encoded

traindata = traindata.drop('GarageType',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[126]:





#We analyse GarageYrBlt... There are no null values

traindata['GarageYrBlt'].isnull().sum(axis = 0)





# In[127]:





traindata['GarageYrBlt'] = traindata['GarageYrBlt'].replace(np.nan, 0, regex=True)





# In[128]:





#We analyse GarageFinish... There are no null values

traindata['GarageFinish'].isnull().sum(axis = 0)





# In[129]:





traindata['GarageYrBlt'] = traindata['GarageYrBlt'].replace(np.nan, 'NoGarage', regex=True)





# In[130]:







#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['GarageYrBlt'] = 'GarageYrBlt-' + traindata['GarageYrBlt'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['GarageYrBlt'])

# Drop column Product as it is now encoded

traindata = traindata.drop('GarageYrBlt',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[131]:





#We analyse GarageCars... There are no null values so we do not need to edit

traindata['GarageCars'].isnull().sum(axis = 0)





# In[132]:





#We analyse GarageArea... There are no null values so we do not need to edit

traindata['GarageArea'].isnull().sum(axis = 0)





# In[133]:





#We analyse GarageQual... There are no null values so we do not need to edit

traindata['GarageQual'].isnull().sum(axis = 0)





# In[134]:





traindata['GarageQual'] = traindata['GarageQual'].replace(np.nan, 'nogarg', regex=True)





# In[135]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['GarageQual'] = 'GarageQual-' + traindata['GarageQual'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['GarageQual'])

# Drop column Product as it is now encoded

traindata = traindata.drop('GarageQual',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[136]:





#Analyse GarageCond

traindata['GarageCond'] = traindata['GarageCond'].replace(np.nan, 'nogarageforcond', regex=True)





# In[137]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['GarageCond'] = 'GarageCond-' + traindata['GarageCond'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['GarageCond'])

# Drop column Product as it is now encoded

traindata = traindata.drop('GarageCond',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[138]:





#We analyse PavedDrive... There are no null values so we do not need to edit

traindata['PavedDrive'].isnull().sum(axis = 0)





# In[139]:





#We need to change the names of the variables in PavedDrive in order to perform OneHotENcoding

traindata['PavedDrive'] = 'PavedDrive-' + traindata['PavedDrive'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['PavedDrive'])

# Drop column Product as it is now encoded

traindata = traindata.drop('PavedDrive',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[140]:





#We analyse GarageQual... There are no null values so we do not need to edit

traindata['WoodDeckSF'].isnull().sum(axis = 0)





# In[141]:





#We analyse OpenPorchSF... There are no null values so we do not need to edit

traindata['OpenPorchSF'].isnull().sum(axis = 0)





# In[142]:





#We analyse OpenPorchSF... There are no null values so we do not need to edit

traindata['EnclosedPorch'].isnull().sum(axis = 0)





# In[143]:





#We analyse 3SsnPorch... There are no null values so we do not need to edit

traindata['3SsnPorch'].isnull().sum(axis = 0)





# In[144]:





#We analyse ScreenPorch... There are no null values so we do not need to edit

traindata['ScreenPorch'].isnull().sum(axis = 0)





# In[145]:





#We analyse ScreenPorch... There are no null values so we do not need to edit

traindata['PoolArea'].isnull().sum(axis = 0)





# In[146]:





#We analyse PoolQC... 

traindata['PoolQC'].isnull().sum(axis = 0)





# In[147]:





traindata['PoolQC'] = traindata['PoolQC'].replace(np.nan, 'Nopool', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['PoolQC'] = 'PoolQC-' + traindata['PoolQC'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['PoolQC'])

# Drop column Product as it is now encoded

traindata = traindata.drop('PoolQC',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[148]:





#We analyse Fence... 

traindata['Fence'] = traindata['Fence'].replace(np.nan, 'Nofence', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['Fence'] = 'Fence-' + traindata['Fence'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Fence'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Fence',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[149]:





#We analyse MiscFeature... 

traindata['MiscFeature'] = traindata['MiscFeature'].replace(np.nan, 'MiscFeature', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['MiscFeature'] = 'MiscFeature-' + traindata['MiscFeature'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['MiscFeature'])

# Drop column Product as it is now encoded

traindata = traindata.drop('MiscFeature',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[150]:





#We analyse MiscVal... There are no null values so we do not need to edit

traindata['MiscVal'].isnull().sum(axis = 0)





# In[151]:





#We analyse MoSold... There are no null values so we do not need to edit

traindata['MoSold'].isnull().sum(axis = 0)





# In[152]:





#We analyse YrSold... There are no null values so we do not need to edit

traindata['YrSold'].isnull().sum(axis = 0)





# In[153]:





#We analyse SaleType... 

traindata['SaleType'] = traindata['SaleType'].replace(np.nan, 'noSaleType', regex=True)

#We need to change the names of the variables in SaleType in order to perform OneHotENcoding

traindata['SaleType'] = 'SaleType-' + traindata['SaleType'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['SaleType'])

# Drop column Product as it is now encoded

traindata = traindata.drop('SaleType',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[154]:





#We analyse SaleCondition... There are no null values so we do not need to edit

traindata['SaleCondition'].isnull().sum(axis = 0)





# In[155]:





traindata['SaleCondition'].unique()





# In[156]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['SaleCondition'])

# Drop column Product as it is now encoded

traindata = traindata.drop('SaleCondition',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[ ]:











# In[ ]:











# In[ ]:











# In[158]:





# We need to analyse- BsmtCond



traindata['BsmtCond'] = traindata['BsmtCond'].replace(np.nan, 'NoBsmtCond', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['BsmtCond'] = 'BsmtCond-' + traindata['BsmtCond'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BsmtCond'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BsmtCond',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[159]:





# We need to analyse- BsmtCond



traindata['BsmtFinType1'] = traindata['BsmtFinType1'].replace(np.nan, 'BsmtFinType1', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['BsmtFinType1'] = 'BsmtFinType1-' + traindata['BsmtFinType1'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['BsmtFinType1'])

# Drop column Product as it is now encoded

traindata = traindata.drop('BsmtFinType1',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[ ]:











# In[161]:





# We need to analyse- Functional



traindata['Functional'] = traindata['Functional'].replace(np.nan, 'Functional', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['Functional'] = 'Functional-' + traindata['Functional'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['Functional'])

# Drop column Product as it is now encoded

traindata = traindata.drop('Functional',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()





# In[162]:





# We need to analyse- Functional



traindata['GarageFinish'] = traindata['GarageFinish'].replace(np.nan, 'GarageFinish', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

traindata['GarageFinish'] = 'GarageFinish-' + traindata['GarageFinish'].astype(str)

traindata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(traindata['GarageFinish'])

# Drop column Product as it is now encoded

traindata = traindata.drop('GarageFinish',axis = 1)

# Join the encoded traindata

traindata = traindata.join(one_hot)

traindata.head()













#Divide into training data and testing data

y=traindata['SalePrice']

x=traindata.drop('SalePrice',axis=1)

#Splitting training and testing data

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.90,test_size=0.10, random_state=0)



##We use different ml models to see which works best.

#Decision Tree regressor

regressor = DecisionTreeRegressor(random_state = 0)  

  

# fit the regressor with X and Y data 

regressor.fit(x_train, y_train) 

y_predi = regressor.predict(x_test) 

mse = mean_squared_error(y_test, y_predi)

r = r2_score(y_test, y_predi)

mae = mean_absolute_error(y_test,y_predi)

rms=np.sqrt(mse)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

print("RMSE:",rms)
#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x_train, y_train)

y_predicted = linearRegressor.predict(x_test)

mse = mean_squared_error(y_test, y_predicted)

r = r2_score(y_test, y_predicted)

mae = mean_absolute_error(y_test,y_predicted)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)



#Polynomial Regression

polynomial_features= PolynomialFeatures(degree=2)

x_poly = polynomial_features.fit_transform(x_train)

x_poly_test = polynomial_features.fit_transform(x_test)

model = LinearRegression()

model.fit(x_poly, y_train)

y_predicted_p = model.predict(x_poly_test)

mse = mean_squared_error(y_test, y_predicted_p)

r = r2_score(y_test, y_predicted_p)

mae = mean_absolute_error(y_test,y_predicted_p)

print("Mean Squared Error:",mse)

print("R score:",r)

print("Mean Absolute Error:",mae)

#!/usr/bin/env python

# coding: utf-8



# In[1]:











# In[2]:





#filename=r'C:\Users\Vibeesh\Desktop\kaggle data\HousePrices\test.csv'

#testdata=pd.read_csv(filename)

#testdata.head()





# In[3]:





##EDA

#Here we see that there are a large number of null values in 

sns.heatmap(testdata.isnull(),yticklabels=False,cbar=False,cmap='viridis')





# In[4]:





## Now we deal with the columns one by one.

#The ID of the house does not influence the sales price. So, we remove them completely.

testdata=testdata.drop('Id',axis=1)

testdata.head()





# In[5]:





#The MSSubClass column is does not need to be processed.



#Now, We check the MSZoning column.

testdata['MSZoning'].unique()





# In[6]:





#The MSZoning column contains categorical variables so One Hot Encoding must be done.

one_hot = pd.get_dummies(testdata['MSZoning'])

# Drop column Product as it is now encoded

testdata = testdata.drop('MSZoning',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[7]:





#The LotFrontage column as seen in the graph, contains many null values. We need to replace the null values by the mean of the entire column



testdata['LotFrontage'].fillna((testdata['LotFrontage'].mean()), inplace=True)  

testdata.head()





# In[8]:





#The ALley column as seen in the graph has too many missing values. Therefore we remove it completely.

testdata=testdata.drop('Alley',axis=1)

testdata.head()





# In[9]:





#Now we analyse the utilities column.

testdata['Utilities'].unique()

testdata['Utilities'] = testdata['Utilities'].replace(np.nan, 'Utilities', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['Utilities'] = 'Utilities-' + testdata['Utilities'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Utilities'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Utilities',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()



# In[10]:





#We see that this column contains categorical variables and must be replaced by numbers 1 and 0-



testdata.head()





# In[11]:





#Now we analyse the LotArea column. .... this column does not contain any null values and can be left as it is

testdata['LotArea'].isnull().sum(axis = 0)





# In[12]:





# We analyse the street column now-

testdata['Street'].unique()





# In[13]:





testdata['Street'].isnull().sum(axis = 0)





# In[14]:





#This column contains categorical variables and must be converted to numerical form-

testdata['Street'] = testdata['Street'].replace('Pave', 0)

testdata['Street'] = testdata['Street'].replace('Grvl', 1)

testdata.head()





# In[15]:





# We analyse the LotShape column now-

testdata['LotShape'].unique()





# In[16]:





testdata['LotShape'].isnull().sum(axis = 0)





# In[17]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['LotShape'])

# Drop column Product as it is now encoded

testdata = testdata.drop('LotShape',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[18]:





#Now we need to analyse the LandContour

testdata['LandContour'].unique()





# In[19]:





testdata['LandContour'].isnull().sum(axis = 0)





# In[20]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['LandContour'])

# Drop column Product as it is now encoded

testdata = testdata.drop('LandContour',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[21]:





#Now we need to analyse the LotConfig

testdata['LotConfig'].unique()





# In[22]:





testdata['LotConfig'].isnull().sum(axis = 0)





# In[23]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['LotConfig'])

# Drop column Product as it is now encoded

testdata = testdata.drop('LotConfig',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[24]:





#Now we need to analyse the LandSlope

testdata['LandSlope'].unique()





# In[25]:





testdata['LandSlope'].isnull().sum(axis = 0)





# In[26]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['LandSlope'])

# Drop column Product as it is now encoded

testdata = testdata.drop('LandSlope',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[27]:





#Now we need to analyse the LandSlope

testdata['Neighborhood'].unique()





# In[28]:





testdata['Neighborhood'].isnull().sum(axis = 0)





# In[29]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Neighborhood'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Neighborhood',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[30]:





#Now we need to analyse the Condition1

testdata['Condition1'].unique()





# In[31]:





testdata['Condition1'].isnull().sum(axis = 0)





# In[32]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Condition1'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Condition1',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[33]:





#Now we need to analyse the Condition2

testdata['Condition2'].unique()





# In[34]:





testdata['Condition2'].isnull().sum(axis = 0)





# In[35]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for Condition1

testdata['Condition2'] = 'condition2-' + testdata['Condition2'].astype(str)

testdata.head()





# In[36]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Condition2'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Condition2',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[37]:





#Now we need to analyse the BldgType

testdata['BldgType'].unique()





# In[38]:





testdata['BldgType'].isnull().sum(axis = 0)





# In[39]:







#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BldgType'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BldgType',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[40]:





#Now we need to analyse the HouseStyle

testdata['HouseStyle'].unique()





# In[41]:





testdata['HouseStyle'].isnull().sum(axis = 0)





# In[42]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['HouseStyle'])

# Drop column Product as it is now encoded

testdata = testdata.drop('HouseStyle',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[43]:





# Now we analyse the Overall Quality- There are no null values so they are used as given.

testdata['OverallQual'].isnull().sum(axis = 0)





# In[44]:





# Now we analyse the Overall Quality- There are no null values so they are used as given.

testdata['OverallCond'].isnull().sum(axis = 0)





# In[45]:





#Now we analyse the year built-There are no null values so they are used as given.

testdata['YearBuilt'].isnull().sum(axis = 0)





# In[46]:





#Now we analyse the year remodelled column-There are no null values so they are used as given.

testdata['YearRemodAdd'].isnull().sum(axis = 0)





# In[47]:





#Now we analyse the RoofStyle column

testdata['RoofStyle'].isnull().sum(axis = 0)





# In[48]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['RoofStyle'])

# Drop column Product as it is now encoded

testdata = testdata.drop('RoofStyle',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[49]:





#Now we analyse the RoofMatl column

testdata['RoofMatl'].isnull().sum(axis = 0)





# In[50]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['RoofMatl'])

# Drop column Product as it is now encoded

testdata = testdata.drop('RoofMatl',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[51]:





#Now we analyse the Exterior1st column

testdata['Exterior1st'].isnull().sum(axis = 0)





# In[52]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Exterior1st'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Exterior1st',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[53]:





#Now we analyse the Exterior2nd column

testdata['Exterior2nd'].isnull().sum(axis = 0)





# In[54]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for Condition1

testdata['Exterior2nd'] = 'Exterior2nd-' + testdata['Exterior2nd'].astype(str)

testdata.head()





# In[55]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Exterior2nd'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Exterior2nd',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[56]:





#Now we analyse the Exterior2nd column

testdata['MasVnrType'].isnull().sum(axis = 0)





# In[57]:





#We replace the null values with "None"

testdata['MasVnrType'] = testdata['MasVnrType'].replace(np.nan, 'None', regex=True)





# In[58]:





testdata['MasVnrType'].isnull().sum(axis = 0)





# In[59]:





testdata['MasVnrType'].unique()





# In[60]:





#We need to change the names of the variables in Condition2 in order to perform OneHotENcoding so that they do not overlap with the OHE done for a previous column1

testdata['MasVnrType'] = 'MasVnrType-' + testdata['MasVnrType'].astype(str)

testdata.head()





# In[61]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['MasVnrType'])

# Drop column Product as it is now encoded

testdata = testdata.drop('MasVnrType',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[62]:





#Now we analyse the Exterior2nd column

testdata['MasVnrArea'].isnull().sum(axis = 0)





# In[63]:





#We replace the null values with "None"

testdata['MasVnrArea'] = testdata['MasVnrArea'].replace(np.nan, 0, regex=True)





# In[64]:





testdata['MasVnrArea'].isnull().sum(axis = 0)





# In[65]:





testdata['ExterQual'].isnull().sum(axis = 0)





# In[66]:





testdata['ExterQual'].unique()





# In[67]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['ExterQual'])

# Drop column Product as it is now encoded

testdata = testdata.drop('ExterQual',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[68]:





#Analyse ExterCond

testdata['ExterCond'].isnull().sum(axis = 0)





# In[69]:





#We need to change the names of the variables in ExterCond in order to perform OneHotENcoding so that they do not overlap with the OHE done for ExterQual

testdata['ExterCond'] = 'xterCond-' + testdata['ExterCond'].astype(str)

testdata.head()





# In[70]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['ExterCond'])

# Drop column Product as it is now encoded

testdata = testdata.drop('ExterCond',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[71]:





#We analyse Foundation

testdata['Foundation'].isnull().sum(axis = 0)





# In[72]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

testdata['Foundation'] = 'Foundation-' + testdata['Foundation'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Foundation'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Foundation',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[73]:





#Now we analysse BsmtQual

testdata['BsmtQual'].value_counts(dropna=False)





# In[74]:





testdata['BsmtQual'] = testdata['BsmtQual'].replace(np.nan, 'None1', regex=True)





# In[75]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

testdata['BsmtQual'] = 'BsmtQual-' + testdata['BsmtQual'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BsmtQual'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BsmtQual',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[76]:





#We analyse BsmtExposure

testdata['BsmtExposure'].isnull().sum(axis = 0)





# In[77]:





testdata['BsmtExposure'] = testdata['BsmtExposure'].replace(np.nan, 'No', regex=True)





# In[78]:





testdata['BsmtExposure'].unique()





# In[79]:





#We need to change the names of the variables in Foundation in order to perform OneHotENcoding

testdata['BsmtExposure'] = 'BsmtExposure-' + testdata['BsmtExposure'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BsmtExposure'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BsmtExposure',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[80]:





#We analyse BsmtFinSF1 ....There is no null values so we dont have to edit it.

testdata['BsmtFinSF1'].isnull().sum(axis = 0)





# In[81]:





#We analyse BsmtFinType2

testdata['BsmtFinType2'].isnull().sum(axis = 0)





# In[82]:





testdata['BsmtFinType2'].value_counts(dropna=False)





# In[83]:





testdata['BsmtFinType2'] = testdata['BsmtFinType2'].replace(np.nan, 'Unf', regex=True)





# In[84]:





testdata['BsmtFinType2'].isnull().sum(axis = 0)





# In[85]:





#We need to change the names of the variables in BsmtFinType2 in order to perform OneHotENcoding

testdata['BsmtFinType2'] = 'BsmtFinType2-' + testdata['BsmtFinType2'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BsmtFinType2'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BsmtFinType2',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[86]:





#We analyse BsmtFinType2...There is no null values so we dont have to edit it.

testdata['BsmtFinSF2'].isnull().sum(axis = 0)





# In[87]:





#We analyse BsmtUnfSF...There is no null values so we dont have to edit it.

testdata['BsmtUnfSF'].isnull().sum(axis = 0)





# In[88]:





#We analyse TotalBsmtSF...There is no null values so we dont have to edit it.

testdata['TotalBsmtSF'].isnull().sum(axis = 0)





# In[89]:





#We analyse Heating...There is no null values 

testdata['Heating'].isnull().sum(axis = 0)





# In[90]:





testdata['Heating'].unique()





# In[91]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Heating'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Heating',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[92]:





#We analyse HeatingQC...There is no null values 

testdata['HeatingQC'].isnull().sum(axis = 0)





# In[93]:





#We need to change the names of the variables in HeatingQC in order to perform OneHotENcoding

testdata['HeatingQC'] = 'HeatingQC-' + testdata['HeatingQC'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['HeatingQC'])

# Drop column Product as it is now encoded

testdata = testdata.drop('HeatingQC',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[94]:





#We analyse CentralAir...There is no null values 

testdata['CentralAir'].isnull().sum(axis = 0)





# In[95]:





testdata['CentralAir'].unique()





# In[96]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['CentralAir'])

# Drop column Product as it is now encoded

testdata = testdata.drop('CentralAir',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[97]:





#We analyse Electrical...

testdata['Electrical'].isnull().sum(axis = 0)





# In[98]:





testdata['Electrical'].unique()





# In[99]:





testdata['Electrical'].value_counts()





# In[100]:





testdata['Electrical'] = testdata['Electrical'].replace(np.nan, 'SBrkr ', regex=True)





# In[101]:





testdata['Electrical'].isnull().sum(axis = 0)





# In[102]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Electrical'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Electrical',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[103]:





#We analyse 1stFlrSF... There are no null values so we do not need to edit them

testdata['1stFlrSF'].isnull().sum(axis = 0)





# In[104]:





#We analyse 2ndFlrSF... There are no null values so we do not need to edit them

testdata['2ndFlrSF'].isnull().sum(axis = 0)





# In[105]:





#We analyse LowQualFinSF... There are no null values so we do not need to edit them

testdata['LowQualFinSF'].isnull().sum(axis = 0)





# In[106]:





testdata['LowQualFinSF'].unique()





# In[107]:





#We analyse GrLivArea... There are no null values so we do not need to edit them

testdata['GrLivArea'].isnull().sum(axis = 0)





# In[108]:





#We analyse BsmtFullBath... There are no null values so we do not need to edit them

testdata['BsmtFullBath'].isnull().sum(axis = 0)





# In[109]:





#We analyse BsmtHalfBath... There are no null values so we do not need to edit them

testdata['BsmtHalfBath'].isnull().sum(axis = 0)





# In[110]:





#We analyse FullBath... There are no null values so we do not need to edit them

testdata['FullBath'].isnull().sum(axis = 0)





# In[111]:





#We analyse HalfBath... There are no null values so we do not need to edit them

testdata['HalfBath'].isnull().sum(axis = 0)





# In[112]:





#We analyse BedroomAbvGr... There are no null values so we do not need to edit them

testdata['BedroomAbvGr'].isnull().sum(axis = 0)





# In[113]:





#We analyse KitchenAbvGr... There are no null values so we do not need to edit them

testdata['KitchenAbvGr'].isnull().sum(axis = 0)





# In[114]:





#We analyse KitchenQual... There are no null values

testdata['KitchenQual'].isnull().sum(axis = 0)





# In[115]:





#We need to change the names of the variables in BsmtFinType2 in order to perform OneHotENcoding

testdata['KitchenQual'] = 'KitchenQual-' + testdata['KitchenQual'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['KitchenQual'])

# Drop column Product as it is now encoded

testdata = testdata.drop('KitchenQual',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[116]:





#We analyse TotRmsAbvGrd... There are no null values

testdata['TotRmsAbvGrd'].isnull().sum(axis = 0)





# In[117]:





#We analyse Functional... There are no null values

testdata['Functional'].isnull().sum(axis = 0)





# In[118]:





#We analyse Fireplaces... There are no null values

testdata['Fireplaces'].isnull().sum(axis = 0)





# In[119]:





#We analyse FireplaceQu... 

testdata['FireplaceQu'].isnull().sum(axis = 0)





# In[120]:





testdata['FireplaceQu'] = testdata['FireplaceQu'].replace(np.nan, 'NoFireplace', regex=True)





# In[121]:





testdata['FireplaceQu'].isnull().sum(axis = 0)





# In[122]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['FireplaceQu'] = 'FireplaceQu-' + testdata['FireplaceQu'].astype(str)

testdata.head()

#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['FireplaceQu'])

# Drop column Product as it is now encoded

testdata = testdata.drop('FireplaceQu',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[123]:





#We analyse GarageType... There are no null values

testdata['GarageType'].isnull().sum(axis = 0)





# In[124]:





testdata['GarageType'] = testdata['GarageType'].replace(np.nan, 'Nocar', regex=True)





# In[125]:







#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['GarageType'] = 'GarageType-' + testdata['GarageType'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['GarageType'])

# Drop column Product as it is now encoded

testdata = testdata.drop('GarageType',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[126]:





#We analyse GarageYrBlt... There are no null values

testdata['GarageYrBlt'].isnull().sum(axis = 0)





# In[127]:





testdata['GarageYrBlt'] = testdata['GarageYrBlt'].replace(np.nan, 0, regex=True)





# In[128]:





#We analyse GarageFinish... There are no null values

testdata['GarageFinish'].isnull().sum(axis = 0)





# In[129]:





testdata['GarageYrBlt'] = testdata['GarageYrBlt'].replace(np.nan, 'NoGarage', regex=True)





# In[130]:







#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['GarageYrBlt'] = 'GarageYrBlt-' + testdata['GarageYrBlt'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['GarageYrBlt'])

# Drop column Product as it is now encoded

testdata = testdata.drop('GarageYrBlt',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[131]:





#We analyse GarageCars... There are no null values so we do not need to edit

testdata['GarageCars'].isnull().sum(axis = 0)





# In[132]:





#We analyse GarageArea... There are no null values so we do not need to edit

testdata['GarageArea'].isnull().sum(axis = 0)





# In[133]:





#We analyse GarageQual... There are no null values so we do not need to edit

testdata['GarageQual'].isnull().sum(axis = 0)





# In[134]:





testdata['GarageQual'] = testdata['GarageQual'].replace(np.nan, 'nogarg', regex=True)





# In[135]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['GarageQual'] = 'GarageQual-' + testdata['GarageQual'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['GarageQual'])

# Drop column Product as it is now encoded

testdata = testdata.drop('GarageQual',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[136]:





#Analyse GarageCond

testdata['GarageCond'] = testdata['GarageCond'].replace(np.nan, 'nogarageforcond', regex=True)





# In[137]:





#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['GarageCond'] = 'GarageCond-' + testdata['GarageCond'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['GarageCond'])

# Drop column Product as it is now encoded

testdata = testdata.drop('GarageCond',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[138]:





#We analyse PavedDrive... There are no null values so we do not need to edit

testdata['PavedDrive'].isnull().sum(axis = 0)





# In[139]:





#We need to change the names of the variables in PavedDrive in order to perform OneHotENcoding

testdata['PavedDrive'] = 'PavedDrive-' + testdata['PavedDrive'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['PavedDrive'])

# Drop column Product as it is now encoded

testdata = testdata.drop('PavedDrive',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[140]:





#We analyse GarageQual... There are no null values so we do not need to edit

testdata['WoodDeckSF'].isnull().sum(axis = 0)





# In[141]:





#We analyse OpenPorchSF... There are no null values so we do not need to edit

testdata['OpenPorchSF'].isnull().sum(axis = 0)





# In[142]:





#We analyse OpenPorchSF... There are no null values so we do not need to edit

testdata['EnclosedPorch'].isnull().sum(axis = 0)





# In[143]:





#We analyse 3SsnPorch... There are no null values so we do not need to edit

testdata['3SsnPorch'].isnull().sum(axis = 0)





# In[144]:





#We analyse ScreenPorch... There are no null values so we do not need to edit

testdata['ScreenPorch'].isnull().sum(axis = 0)





# In[145]:





#We analyse ScreenPorch... There are no null values so we do not need to edit

testdata['PoolArea'].isnull().sum(axis = 0)





# In[146]:





#We analyse PoolQC... 

testdata['PoolQC'].isnull().sum(axis = 0)





# In[147]:





testdata['PoolQC'] = testdata['PoolQC'].replace(np.nan, 'Nopool', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['PoolQC'] = 'PoolQC-' + testdata['PoolQC'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['PoolQC'])

# Drop column Product as it is now encoded

testdata = testdata.drop('PoolQC',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[148]:





#We analyse Fence... 

testdata['Fence'] = testdata['Fence'].replace(np.nan, 'Nofence', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['Fence'] = 'Fence-' + testdata['Fence'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Fence'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Fence',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[149]:





#We analyse MiscFeature... 

testdata['MiscFeature'] = testdata['MiscFeature'].replace(np.nan, 'MiscFeature', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['MiscFeature'] = 'MiscFeature-' + testdata['MiscFeature'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['MiscFeature'])

# Drop column Product as it is now encoded

testdata = testdata.drop('MiscFeature',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[150]:





#We analyse MiscVal... There are no null values so we do not need to edit

testdata['MiscVal'].isnull().sum(axis = 0)





# In[151]:





#We analyse MoSold... There are no null values so we do not need to edit

testdata['MoSold'].isnull().sum(axis = 0)





# In[152]:





#We analyse YrSold... There are no null values so we do not need to edit

testdata['YrSold'].isnull().sum(axis = 0)





# In[153]:





#We analyse SaleType... 

testdata['SaleType'] = testdata['SaleType'].replace(np.nan, 'noSaleType', regex=True)

#We need to change the names of the variables in SaleType in order to perform OneHotENcoding

testdata['SaleType'] = 'SaleType-' + testdata['SaleType'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['SaleType'])

# Drop column Product as it is now encoded

testdata = testdata.drop('SaleType',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[154]:





#We analyse SaleCondition... There are no null values so we do not need to edit

testdata['SaleCondition'].isnull().sum(axis = 0)





# In[155]:





testdata['SaleCondition'].unique()





# In[156]:





#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['SaleCondition'])

# Drop column Product as it is now encoded

testdata = testdata.drop('SaleCondition',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[ ]:











# In[ ]:











# In[ ]:











# In[158]:





# We need to analyse- BsmtCond



testdata['BsmtCond'] = testdata['BsmtCond'].replace(np.nan, 'NoBsmtCond', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['BsmtCond'] = 'BsmtCond-' + testdata['BsmtCond'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BsmtCond'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BsmtCond',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[159]:





# We need to analyse- BsmtCond



testdata['BsmtFinType1'] = testdata['BsmtFinType1'].replace(np.nan, 'BsmtFinType1', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['BsmtFinType1'] = 'BsmtFinType1-' + testdata['BsmtFinType1'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['BsmtFinType1'])

# Drop column Product as it is now encoded

testdata = testdata.drop('BsmtFinType1',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[ ]:











# In[161]:





# We need to analyse- Functional



testdata['Functional'] = testdata['Functional'].replace(np.nan, 'Functional', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['Functional'] = 'Functional-' + testdata['Functional'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['Functional'])

# Drop column Product as it is now encoded

testdata = testdata.drop('Functional',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()





# In[162]:





# We need to analyse- Functional



testdata['GarageFinish'] = testdata['GarageFinish'].replace(np.nan, 'GarageFinish', regex=True)

#We need to change the names of the variables in FireplaceQu in order to perform OneHotENcoding

testdata['GarageFinish'] = 'GarageFinish-' + testdata['GarageFinish'].astype(str)

testdata.head()



#This column contains categorical variables and must be converted to numerical form using One Hot Encoding-

one_hot = pd.get_dummies(testdata['GarageFinish'])

# Drop column Product as it is now encoded

testdata = testdata.drop('GarageFinish',axis = 1)

# Join the encoded testdata

testdata = testdata.join(one_hot)

testdata.head()


a=list(testdata.columns.values)

b=list(x_test.columns.values)

r2 = [item for item in b if item not in a]  

r3=[items for items in a if items not in b] 

ff= testdata

xx=traindata

xx.drop(xx[r2], axis = 1) 

ff.drop(ff[r3], axis = 1) 
##Now we fit the ml model to the test data to make the prediction

#Linear Regression

linearRegressor = LinearRegression()

linearRegressor.fit(x, y)

y_predicted = linearRegressor.predict(x_test)
predictionlist=y_predicted.tolist()

Passengerid=testdatacopy['Id'].tolist() 

output=pd.DataFrame(list(zip(Passengerid, predictionlist)),

              columns=['Id','SalePrice'])

output.head()



#Finally, we convert the output to a csv file in order to make the submission

output.to_csv('my_submission(Houseprices).csv', index=False)