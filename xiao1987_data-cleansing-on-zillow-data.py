# Start with importing essentials

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_train = pd.read_csv('../input/train_2016_v2.csv')

df_property = pd.read_csv('../input/properties_2016.csv')
df_train.shape
df_train.head()
df_property.shape
df_property.head()
df_train = df_train.merge(df_property, how='left', on='parcelid')
df_train.shape
df_train.drop(['parcelid','transactiondate'],axis=1,inplace=True)
df_train.shape
df_train.head()
missing_df = df_train.isna().sum().to_frame().reset_index()  #create dataframe with missing data information of each column

missing_df.columns = ['column name','#missing values'] # rename column names 

missing_df = missing_df.loc[missing_df['#missing values']>0] # only keep those who have more than 1 missing value

missing_df.sort_values(by='#missing values',ascending=True,inplace=True) # sort missing value

missing_df 
missing_df.shape
missing_df.plot.barh(x='column name', y = '#missing values', figsize=(15,20))
plt.figure(figsize=(20,30))

plt.barh(missing_df['column name'],missing_df['#missing values'],log=True)
catcols = ['airconditioningtypeid','architecturalstyletypeid','buildingqualitytypeid','buildingclasstypeid','decktypeid','fips','hashottuborspa','heatingorsystemtypeid','pooltypeid10','pooltypeid2','pooltypeid7','propertycountylandusecode','propertylandusetypeid','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood','regionidzip','storytypeid','typeconstructiontypeid','yearbuilt','taxdelinquencyflag']

numcols = [x for x in df_train.columns if x not in catcols]
numcols
plt.figure(figsize = (14,12))

sns.heatmap(data=df_train[numcols].corr())

plt.show()
dropcols = []

dropcols.append('finishedsquarefeet12')

dropcols.append('finishedsquarefeet13')

dropcols.append('finishedsquarefeet15')

dropcols.append('finishedsquarefeet6')

dropcols.append('finishedsquarefeet50')

dropcols.append('calculatedbathnbr')

dropcols.append('fullbathcnt')
df_train['hashottuborspa'].isnull().sum()
df_train['hashottuborspa']=df_train['hashottuborspa'].fillna('None', inplace=True)

# index = df_train.hashottuborspa.isnull()

# df_train.loc[index,'hashottuborspa'] = 'None'
df_train[df_train.pooltypeid10.isnull()

         |df_train.pooltypeid2.isnull()

         |df_train.pooltypeid7.isnull()].hashottuborspa.fillna('None', inplace =True)
df_train.loc[0:5,['hashottuborspa']]
df_train['taxdelinquencyflag'] = df_train['taxdelinquencyflag'].fillna("doesn't exist",inplace=True)
df_train.loc[0:5,['taxdelinquencyflag']]
df_train.loc[df_train.garagecarcnt.isnull(),'garagetotalsqft']=0

# setting value for items matching condition 
df_train.loc[0:10,['garagecarcnt','garagetotalsqft']]
poolsizesum_median = df_train.loc[df_train['poolcnt']>0,'poolsizesum'].median()

poolsizesum_median
df_train.loc[(df_train['poolcnt']>0) & (df_train['poolsizesum'].isnull()),'poolsizesum']=poolsizesum_median

df_train.loc[(df_train['poolcnt']==0),'poolsizesum']=0
df_train.loc[:5,'poolsizesum']
df_train.fireplaceflag.value_counts()
df_train.fireplaceflag='No'
df_train.fireplaceflag.value_counts()
df_train.loc[df_train['fireplacecnt']>0,'fireplaceflag']='Yes'

df_train.loc[df_train['fireplacecnt'].isnull(),'fireplaceflag']=0
df_train.fireplaceflag.value_counts()
df_train.airconditioningtypeid.fillna(1,inplace= True)
df_train.heatingorsystemtypeid.fillna(2,inplace= True)
missingvalues_prop = (df_train.isnull().sum()/len(df_train)).reset_index()

missingvalues_prop.columns = ['name','proportion']

missingvalues_prop = missingvalues_prop.sort_values (by = 'proportion', ascending=False)

print(missingvalues_prop)

missingvaluescols = missingvalues_prop [missingvalues_prop['proportion']>0.97].field.tolist()

dropcols = dropcols + missingvaluescols

df_train = df_train.drop (dropcols,axis=1)
a = np.array([True,False])

print(~a)
from sklearn import neighbors

from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



## Works on categorical feature

def fillna_knn( df, base, target, fraction = 1, threshold = 10, n_neighbors = 5 ):

    assert isinstance( base , list ) or isinstance( base , np.ndarray ) and isinstance( target, str ) 

    whole = [ target ] + base

    

    miss = df[target].isnull()

    notmiss = ~miss 

    nummiss = miss.sum()

    

    enc = OneHotEncoder()

    X_target = df.loc[ notmiss, whole ].sample( frac = fraction )

    

    enc.fit( X_target[ target ].unique().reshape( (-1,1) ) )

    

    Y = enc.transform( X_target[ target ].values.reshape((-1,1)) ).toarray()

    X = X_target[ base  ]

    

    print( 'fitting' )

    n_neighbors = n_neighbors

    clf = neighbors.KNeighborsClassifier( n_neighbors, weights = 'uniform' )

    clf.fit( X, Y )

    

    print( 'the shape of active features: ' ,enc.active_features_.shape )

    

    print( 'predicting' )

    Z = clf.predict(df.loc[miss, base])

    

    numunperdicted = Z[:,0].sum()

    if numunperdicted / nummiss *100 < threshold :

        print( 'writing result to df' )    

        df.loc[ miss, target ]  = np.dot( Z , enc.active_features_ )

        print( 'num of unperdictable data: ', numunperdicted )

        return enc

    else:

        print( 'out of threshold: {}% > {}%'.format( numunperdicted / nummiss *100 , threshold ) )



#function to deal with variables that are actually string/categories

def zoningcode2int( df, target ):

    storenull = df[ target ].isnull()

    enc = LabelEncoder( )

    df[ target ] = df[ target ].astype( str )



    print('fit and transform')

    df[ target ]= enc.fit_transform( df[ target ].values )

    print( 'num of categories: ', enc.classes_.shape  )

    df.loc[ storenull, target ] = np.nan

    print('recover the nan value')

    return enc



### Example: 

### If you want to impute buildingqualitytypeid with geological information:

"""

fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'buildingqualitytypeid', fraction = 0.15, n_neighbors = 1 )

"""



## Works on regression

def fillna_knn_reg( df, base, target, n_neighbors = 5 ):

    cols = base + [target]

    X_train = df[cols]

    scaler = StandardScaler(with_mean=True, with_std=True).fit(X_train[base].values.reshape(-1, 1))

    rescaledX = scaler.transform(X_train[base].values.reshape(-1, 1))



    X_train = rescaledX[df[target].notnull()]

    Y_train = df.loc[df[target].notnull(),target].values.reshape(-1, 1)



    knn = KNeighborsRegressor(n_neighbors, n_jobs = -1)    

    # fitting the model

    knn.fit(X_train, Y_train)

    # predict the response

    X_test = rescaledX[df[target].isnull()]

    pred = knn.predict(X_test)

    df.loc[df_train[target].isnull(),target] = pred

    return
df_train.columns.values
df_train.loc[:,['latitude','longitude']]
df_train.loc[:,['latitude','longitude']].isnull().sum()
fillna_knn( df = df_train,

                  base = [ 'latitude', 'longitude' ] ,

                  target = 'regionidcity', fraction = 0.15, n_neighbors = 1 )