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
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.neighbors import KNeighborsRegressor



df= pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

dft= pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
df.head()
print(df.shape)

print(dft.shape)
# We took out ID of test file as it will help us map prices of our pridiction for the submission. 

# We took out SalePrice because we added it as the the target for our label.

# Also, IDs didnt quite help adding more value to our features. 



a= df['Id']

b= dft['Id']

y= df['SalePrice']

df.drop({'Id'},axis=1,  inplace= True)

dft.drop({'Id'},axis=1, inplace= True)

df.drop({'SalePrice'}, axis=1, inplace= True)
cat_train= df.select_dtypes(include=['object'])

cat_test= dft.select_dtypes(include=['object'])
print(cat_train.shape)

print(cat_test.shape)
df.isna().sum().sort_values(ascending= False)[:15]
dft.isna().sum().sort_values(ascending= False)[:15]
# As most of them are missing in these 5 features and none of them are quentessential in predicting house prices; Like if i am getting a good deal

# of a house, i would mostly not be concerned whether it has a fence or not. 



df.drop({'PoolQC', 'Alley', 'FireplaceQu', 'GarageYrBlt', 'Fence'}, axis= 1, inplace= True)

dft.drop({'PoolQC', 'Alley', 'FireplaceQu', 'GarageYrBlt', 'Fence'}, axis= 1, inplace= True)



# For training set



for i in range(len(df['MiscFeature'])):

  if df['MiscFeature'][i]== 'Shed':

   df['MiscFeature'][i] = 1

  elif df['MiscFeature'][i] == 'TenC':

   df['MiscFeature'][i] = 2

  else:

   df['MiscFeature'][i] = 0



# For test set



for i in range(len(dft['MiscFeature'])):

  if dft['MiscFeature'][i]== 'Shed':

   dft['MiscFeature'][i] = 1

  elif dft['MiscFeature'][i] == 'TenC':

   dft['MiscFeature'][i] = 2

  else:

   dft['MiscFeature'][i] = 0
print(df['OverallCond'].groupby(df['GarageCond']).median())

print(df['OverallCond'].groupby(df['GarageCond']).count())
# For training set



df['GarageCond'].fillna(0, inplace= True)

for i in range(len(df['GarageCond'])):

  if df['GarageCond'][i] == 0:

    if df['OverallCond'][i] <= 5:

      df['GarageCond'][i] = 'TA'

    elif df['OverallCond'][i] >6:

      df['GarageCond'][i] = 'Gd'

    elif df['OverallCond'][i] == 6:

      df['GarageCond'][i] = 'Fa'
# For test set



dft['GarageCond'].fillna(0, inplace= True)

for i in range(len(dft['GarageCond'])):

  if dft['GarageCond'][i] == 0:

    if dft['OverallCond'][i] <= 5:

      dft['GarageCond'][i] = 'TA'

    elif dft['OverallCond'][i] >6:

      dft['GarageCond'][i] = 'Gd'

    elif dft['OverallCond'][i] == 6:

      dft['GarageCond'][i] = 'Fa'
df['GarageQual'].groupby(df['GarageCond']).value_counts()
df['GarageQual'].fillna(0, inplace= True)

dft['GarageQual'].fillna(0, inplace= True)



for i in range(len(df['GarageQual'])):

    if df['GarageQual'][i] == 0:

        df['GarageQual'][i] = df['GarageCond'][i]



for i in range(len(dft['GarageQual'])):

    if dft['GarageQual'][i] == 0:

        dft['GarageQual'][i] = dft['GarageCond'][i]
df['GarageArea'].groupby(df['GarageFinish']).mean()



# we can see that larger garages tend to be more finished. So,
df['GarageFinish'].fillna(0, inplace= True)

dft['GarageFinish'].fillna(0, inplace= True)



for i in range(len(df)):

  if df['GarageFinish'][i] == 0:

    if df['GarageArea'][i] <= 500:

      df['GarageFinish'][i] = 'Unf'

    elif 500 < df['GarageArea'][i] <= 555:

      df['GarageFinish'][i] = 'RFn'

    else:

      df['GarageFinish'][i] = 'Fin'

    

for i in range(len(dft)):

  if dft['GarageFinish'][i] == 0:

    if dft['GarageArea'][i] <= 500:

      dft['GarageFinish'][i] = 'Unf'

    elif 500 < dft['GarageArea'][i] <= 555:

      dft['GarageFinish'][i] = 'RFn'

    else:

      dft['GarageFinish'][i] = 'Fin'
print(df['GarageType'].value_counts())



print(df['GarageArea'].groupby(df['GarageType']).mean())
df['GarageType'].fillna(0, inplace= True)

dft['GarageType'].fillna(0, inplace= True)





for i in range(len(df['GarageType'])):

  if df['GarageType'][i] == 0:

    if df['GarageArea'][i] > 550:

      df['GarageType'][i] = '2Types'

    elif 400 < df['GarageArea'][i] <= 455:

      df['GarageType'][i] = 'Detchd'

    else:

      df['GarageType'][i] = 'BuiltIn'

    

for i in range(len(dft['GarageType'])):

  if dft['GarageType'][i] == 0:

    if dft['GarageArea'][i] > 550:

      dft['GarageType'][i] = '2Types'

    elif 400 < dft['GarageArea'][i] <= 455:

      dft['GarageType'][i] = 'Detchd'

    else:

      dft['GarageType'][i] = 'BuiltIn'
df['BsmtCond'].fillna(0, inplace= True)

dft['BsmtCond'].fillna(0, inplace= True)



for i in range(len(df['BsmtCond'])):

  if df['BsmtCond'][i] == 0:

    if 1000 < df['TotalBsmtSF'][i] <= 1100:

      df['BsmtCond'][i] = 'TA'

    elif df['TotalBsmtSF'][i] < 840:

      df['BsmtCond'][i] = 'Po'

    else:

      df['BsmtCond'][i] = 'Gd'

    

for i in range(len(dft['BsmtCond'])):

  if dft['BsmtCond'][i] == 0:

    if 1000 < dft['TotalBsmtSF'][i] <= 1100:

      dft['BsmtCond'][i] = 'TA'

    elif dft['TotalBsmtSF'][i] < 840:

      dft['BsmtCond'][i] = 'Po'

    else:

      dft['BsmtCond'][i] = 'Gd'
df['TotalBsmtSF'].groupby(df['BsmtExposure']).median()
df['BsmtExposure'].fillna(0, inplace= True)

dft['BsmtExposure'].fillna(0, inplace= True)



for i in range(len(df['BsmtExposure'])):

  if df['BsmtExposure'][i] == 0:

    if df['TotalBsmtSF'][i] <= 1000:

      df['BsmtExposure'][i] = 'No'

    else:

      df['BsmtExposure'][i] = 'Av'

    

for i in range(len(dft['BsmtExposure'])):

  if dft['BsmtExposure'][i] == 0:

    if dft['TotalBsmtSF'][i] <= 1000:

      dft['BsmtExposure'][i] = 'No'

    else:

      dft['BsmtExposure'][i] = 'Av'
df.drop({'BsmtFinType2'}, axis=1, inplace= True)



dft.drop({'BsmtFinType2'}, axis=1, inplace= True)
df['TotalBsmtSF'].groupby(df['BsmtQual']).median()
df['BsmtQual'].fillna(0, inplace= True)

dft['BsmtQual'].fillna(0, inplace= True)





for i in range(len(df['BsmtQual'])):

  if df['BsmtQual'][i] == 0:

    if df['TotalBsmtSF'][i] <= 1000:

      df['BsmtQual'][i] = 'TA'

    elif 1000 < df['BsmtQual'][i] < 1200:

      df['BsmtQual'][i] = 'Gd'

    else:

      df['BsmtQual'][i] = 'EX'

    

for i in range(len(dft['BsmtQual'])):

  if dft['BsmtQual'][i] == 0:

    if dft['TotalBsmtSF'][i] <= 1000:

      dft['BsmtQual'][i] = 'TA'

    elif 1000 < dft['BsmtQual'][i] < 1200:

      dft['BsmtQual'][i] = 'Gd'

    else:

      dft['BsmtQual'][i] = 'EX'
df['BsmtFinSF1'].groupby(df['BsmtFinType1']).mean()
df['BsmtFinType1'].fillna(0, inplace= True)

dft['BsmtFinType1'].fillna(0, inplace= True)



for i in range(len(df['BsmtFinType1'])):

  if df['BsmtFinType1'][i] == 0:

    if 350 < df['BsmtFinSF1'][i] <= 550:

      df['BsmtFinType1'][i] = 'Rec'

    elif 550 < df['BsmtFinSF1'][i] < 620:

      df['BsmtFinType1'][i] = 'ALQ'

    elif 620< df['BsmtFinSF1'][i] <= 850:

      df['BsmtFinType1'][i] = 'GLQ'

    else:

      df['BsmtFinType1'][i] = 'Unf'

    

for i in range(len(dft['BsmtFinType1'])):

  if dft['BsmtFinType1'][i] == 0:

    if 350 < dft['BsmtFinSF1'][i] <= 550:

      dft['BsmtFinType1'][i] = 'Rec'

    elif 550 < dft['BsmtFinSF1'][i] < 620:

      dft['BsmtFinType1'][i] = 'ALQ'

    elif 620< dft['BsmtFinSF1'][i] <= 850:

      dft['BsmtFinType1'][i] = 'GLQ'

    else:

      dft['BsmtFinType1'][i] = 'Unf'
df['MasVnrType'].fillna(df['MasVnrType'].mode()[0], inplace= True)



dft['MasVnrType'].fillna(dft['MasVnrType'].mode()[0], inplace= True)
df.fillna(0, inplace= True)

dft.fillna(0, inplace= True)
df_tr= pd.get_dummies(df)

df_ts= pd.get_dummies(dft)
print(df_tr.shape)

print(df_ts.shape)
for i in df_tr.columns:

    if i not in df_ts.columns:

        df_tr.drop({i},axis= 1, inplace= True)

        

for i in df_ts.columns:

    if i not in df_tr.columns:

        df_ts.drop({i},axis= 1, inplace= True)
X_train, X_test, y_train, y_test= train_test_split(df_tr, y, random_state= 42)



reg= GradientBoostingRegressor()



reg.fit(df_tr, y)
reg.score(X_test, y_test)
a= reg.predict(df_ts)
a= pd.DataFrame({'Id': b, 'SalePrice':a})
a.set_index('Id', inplace= True)
a.head()
a.to_csv('prices_these_days.csv')



# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "prices_these_days.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a link to download the dataframe

create_download_link(a)