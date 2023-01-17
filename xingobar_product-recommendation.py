

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
limit_row = 6000000

train = pd.read_csv('../input/train_ver2.csv',nrows=limit_row)

unique_ids = train.ncodpers.unique()

unique_ids = unique_ids.sample(n=1e4)

train = train[train.ncodpers.isin(unique_ids)]
train.head()
train.isnull().any()
train.describe()
train['fecha_dato'] = pd.to_datetime(train['fecha_dato'],format='%Y-%m-%d')

train['fecha_alta'] = pd.to_datetime(train['fecha_alta'],format='%Y-%m-%d')

train['month'] = train['fecha_dato'].apply(lambda x:x.month)

train['age'] = pd.to_numeric(train['age'],errors='coerce')
train.isnull().any()
fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(train['age'].dropna(),kde=False,ax=ax,color='#ffa726')

plt.title('Age Distribution')

plt.ylabel('Freq')

plt.xlabel('Age')
train.loc[train.age < 18,"age"]  = train.loc[(train.age >= 18) & (train.age <= 30),"age"].mean(skipna=True)

train.loc[train.age > 100,"age"] = train.loc[(train.age >= 30) & (train.age <= 100),"age"].mean(skipna=True)

train['age'].fillna(train['age'].mean(),inplace=True)

train['age'] = train['age'].astype(int)
sns.set_style('whitegrid')

fig,ax = plt.subplots(figsize=(8,6))

sns.distplot(train['age'],kde=False,ax=ax)

plt.title('Age Distribution')

plt.xlabel('Age')

plt.ylabel('Freq')
train['ind_nuevo'].isnull().sum()
month_active = train.loc[train['ind_nuevo'].isnull(),:].groupby('ncodpers',sort=False).size()

month_active.max()
train['ind_nuevo'].fillna(1,inplace=True)
train['antiguedad'] = pd.to_numeric(train['antiguedad'],errors='coerce')

np.sum(train['antiguedad'].isnull())
train.loc[train['antiguedad'].isnull(),'ind_nuevo'].describe()
train.loc[train['antiguedad'].isnull(),'antiguedad'] = train['antiguedad'].min()

train.loc[train['antiguedad'] <0 , 'antiguedad']  = 0 
dates = train.loc[:,'fecha_alta'].sort_values().reset_index()

median_date = int(np.median(dates.index.values))

train.loc[train['fecha_alta'].isnull(),'fecha_alta'] = train.loc[median_date,'fecha_alta']

train['fecha_alta'].describe()
train['indrel'].value_counts()
train['indrel'].fillna(1,inplace=True)
train.drop(['tipodom','cod_prov'],axis=1,inplace=True)
train.isnull().any()
train['ind_actividad_cliente'].isnull().sum()
train['nomprov'].unique()
train.loc[train.nomprov=="CORU\xc3\x91A, A","nomprov"] = "CORUNA, A"
train.loc[train.nomprov.isnull(),'nomprov']='Unknown'
train.renta.isnull().sum()
train['ind_nomina_ult1'].value_counts()
train['ind_nomina_ult1'].fillna(0,inplace=True)

train['ind_nom_pens_ult1'].fillna(0,inplace=True)
train['indfall'].value_counts()
train['indfall'].fillna('N',inplace=True)
train['tiprel_1mes'].value_counts()
train['tiprel_1mes'].fillna('I',inplace=True)

train['tiprel_1mes'] = train['tiprel_1mes'].astype('category')
train['indrel_1mes'].value_counts()
map_dict = { 1.0  : "1",

            "1.0" : "1",

            "1"   : "1",

            "3.0" : "3",

            "P"   : "P",

            3.0   : "3",

            2.0   : "2",

            "3"   : "3",

            "2.0" : "2",

            "4.0" : "4",

            "4"   : "4",

            "2"   : "2"}
train['indrel_1mes'].fillna('P',inplace=True)

train['indrel_1mes'] = train['indrel_1mes'].apply(lambda x:map_dict.get(x,x))

train['indrel_1mes'] = train['indrel_1mes'].astype('category')
string_data = train.select_dtypes(include=['object'])

missing_columns = [col for col in string_data if string_data[col].isnull().any()]

del string_data
unknown_col = [col for col in missing_columns if col not in ["indfall","tiprel_1mes","indrel_1mes"]]

for col in unknown_col:

    train.loc[train[col].isnull(),col] = 'Unknown'
train.isnull().any()
