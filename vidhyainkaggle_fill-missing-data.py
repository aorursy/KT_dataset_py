# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_2016_v2.csv')

sample = pd.read_csv('../input/sample_submission.csv')

prop = pd.read_csv('../input/properties_2016.csv')
print (train.shape, sample.shape, prop.shape)
dd = pd.read_excel('../input/zillow_data_dictionary.xlsx')
#merge train and properties

x_train = train.merge(prop, how = 'left', on = 'parcelid')
x_train.shape
xtrain = x_train.drop(['calculatedbathnbr',

                       'finishedsquarefeet50',

                       'finishedsquarefeet12',

                       'finishedsquarefeet13',

                       'finishedsquarefeet15',

                       'finishedsquarefeet6',

                       'censustractandblock',

                       'parcelid',

                       'logerror',

                       'transactiondate',

                       'propertyzoningdesc',

                       'propertycountylandusecode',

                       'fireplaceflag',

                       'hashottuborspa',

                       'taxdelinquencyflag',

                        'taxdelinquencyyear'

                        ], axis=1)
xtrain.shape
xtrain.isnull().sum()
m = round(np.median(xtrain['architecturalstyletypeid'].dropna()))

xtrain['architecturalstyletypeid'] = xtrain['architecturalstyletypeid'].fillna(m)



m = round(np.median(xtrain['calculatedfinishedsquarefeet'].dropna()))

xtrain['calculatedfinishedsquarefeet'] = xtrain['calculatedfinishedsquarefeet'].fillna(m)



m = round(xtrain.loc[xtrain['basementsqft'].isnull(),'calculatedfinishedsquarefeet']/5)

xtrain.loc[xtrain['basementsqft'].isnull()==True,'basementsqft'] = m



m =round(np.median(xtrain['buildingclasstypeid'].dropna()))

xtrain.loc[xtrain['buildingclasstypeid'].isnull(),'buildingclasstypeid'] = m



m=round(np.median(xtrain['buildingqualitytypeid'].dropna()))

xtrain.loc[xtrain['buildingqualitytypeid'].isnull(),'buildingqualitytypeid'] = m



m = round(np.median(xtrain['decktypeid'].dropna()))

xtrain.loc[xtrain['decktypeid'].isnull(),'decktypeid'] = m



m = round(np.median(xtrain['finishedfloor1squarefeet'].dropna()))

xtrain.loc[xtrain['finishedfloor1squarefeet'].isnull(),'finishedfloor1squarefeet'] = m



m = round(np.median(xtrain['fireplacecnt'].dropna()))

xtrain.loc[xtrain['fireplacecnt'].isnull(),'fireplacecnt'] = m



m = round(np.median(xtrain['garagecarcnt'].dropna()))

xtrain.loc[xtrain['garagecarcnt'].isnull(),'garagecarcnt'] = m



m = round(np.median(xtrain['garagetotalsqft'].dropna()))

xtrain.loc[xtrain['garagetotalsqft'].isnull(),'garagetotalsqft'] = m



m = round(np.median(xtrain['heatingorsystemtypeid'].dropna()))

xtrain.loc[xtrain['heatingorsystemtypeid'].isnull(),'heatingorsystemtypeid'] = m



m = round(np.median(xtrain['lotsizesquarefeet'].dropna()))

xtrain.loc[xtrain['lotsizesquarefeet'].isnull(),'lotsizesquarefeet'] = m



m = round(np.median(xtrain['airconditioningtypeid'].dropna()))

xtrain.loc[xtrain['airconditioningtypeid'].isnull(), 'airconditioningtypeid'] = m



xtrain.loc[(xtrain['structuretaxvaluedollarcnt']== xtrain['taxvaluedollarcnt'])&(xtrain['landtaxvaluedollarcnt'].isnull() == True), ['landtaxvaluedollarcnt']] = 0.0

xtrain.loc[(xtrain['landtaxvaluedollarcnt']== xtrain['taxvaluedollarcnt'])&(xtrain['structuretaxvaluedollarcnt'].isnull() == True), ['structuretaxvaluedollarcnt']] = 0.0

xtrain.loc[(xtrain['taxvaluedollarcnt'].isnull()==False) & (xtrain['taxamount'].isnull()==True),['taxamount']] = xtrain.loc[(xtrain['taxvaluedollarcnt'].isnull()==False) & (xtrain['taxamount'].isnull()==True),['taxvaluedollarcnt']] / 40

m = round(np.median(xtrain['taxamount'].dropna()))

xtrain.loc[xtrain['taxamount'].isnull(), 'taxamount'] = m

rec = (xtrain.loc[(xtrain['structuretaxvaluedollarcnt'].isnull() == True) & (xtrain['landtaxvaluedollarcnt'].isnull()==True) & (xtrain['taxvaluedollarcnt'].isnull()==True),['taxamount']] * 40 )

xtrain.loc[(xtrain['structuretaxvaluedollarcnt'].isnull() == True) & (xtrain['landtaxvaluedollarcnt'].isnull()==True) & (xtrain['taxvaluedollarcnt'].isnull()==True),['taxvaluedollarcnt']] = rec.values

print(rec.values)

rec = (xtrain.loc[(xtrain['structuretaxvaluedollarcnt'].isnull() == True) & 

          (xtrain['landtaxvaluedollarcnt'].isnull()==True),

          ['taxvaluedollarcnt']]) / 2

print (rec)

xtrain.loc[(xtrain['structuretaxvaluedollarcnt'].isnull() == True) & 

          (xtrain['landtaxvaluedollarcnt'].isnull()==True),

          ['structuretaxvaluedollarcnt','landtaxvaluedollarcnt',]] = rec.values







m = 2.5

m = round(np.median(xtrain['fullbathcnt'].dropna()))

xtrain.loc[xtrain['fullbathcnt'].isnull(), 'fullbathcnt'] = m

print (m)



m = round(np.median(xtrain['threequarterbathnbr'].dropna()))

xtrain.loc[xtrain['threequarterbathnbr'].isnull(), 'threequarterbathnbr'] = m

print (m)



m = round(np.median(xtrain['lotsizesquarefeet'].dropna()))

xtrain.loc[xtrain['lotsizesquarefeet'].isnull(), 'lotsizesquarefeet'] = m

print(m)
xtrain.isnull().sum()
ytrain = x_train['logerror']

dtrain = xgb.DMatrix(xtrain, ytrain)

ymean = np.mean(ytrain)

# xgboost params

xgb_params = {

    'eta': 0.037,

    'max_depth': 5,

    'subsample': 0.80,

    'objective': 'reg:linear',

    'eval_metric': 'mae',

    'lambda': 0.8,   

    'alpha': 0.4, 

    'base_score': ymean,

    'silent': 0

}



model = xgb.train(xgb_params,

                 dtrain,

                 num_boost_round = 242)
train_columns = xtrain.columns

#Prepare test data

#merge sample file and properties file

sample['parcelid'] = sample['ParcelId']

sample_prop = sample.merge(prop, how ='left', on ='parcelid')

xtest = sample_prop[train_columns]

xtest.dtypes[xtest.dtypes == object].index.values
m = round(np.median(xtest['architecturalstyletypeid'].dropna()))

xtest['architecturalstyletypeid'] = xtest['architecturalstyletypeid'].fillna(m)



m = round(np.median(xtest['calculatedfinishedsquarefeet'].dropna()))

xtest['calculatedfinishedsquarefeet'] = xtest['calculatedfinishedsquarefeet'].fillna(m)



m = round(xtest.loc[xtest['basementsqft'].isnull(),'calculatedfinishedsquarefeet']/5)

xtest.loc[xtest['basementsqft'].isnull()==True,'basementsqft'] = m



m =round(np.median(xtest['buildingclasstypeid'].dropna()))

xtest.loc[xtest['buildingclasstypeid'].isnull(),'buildingclasstypeid'] = m



m=round(np.median(xtest['buildingqualitytypeid'].dropna()))

xtest.loc[xtest['buildingqualitytypeid'].isnull(),'buildingqualitytypeid'] = m



m = round(np.median(xtest['decktypeid'].dropna()))

xtest.loc[xtest['decktypeid'].isnull(),'decktypeid'] = m



m = round(np.median(xtest['finishedfloor1squarefeet'].dropna()))

xtest.loc[xtest['finishedfloor1squarefeet'].isnull(),'finishedfloor1squarefeet'] = m



m = round(np.median(xtest['fireplacecnt'].dropna()))

xtest.loc[xtest['fireplacecnt'].isnull(),'fireplacecnt'] = m



m = round(np.median(xtest['garagecarcnt'].dropna()))

xtest.loc[xtest['garagecarcnt'].isnull(),'garagecarcnt'] = m



m = round(np.median(xtest['garagetotalsqft'].dropna()))

xtest.loc[xtest['garagetotalsqft'].isnull(),'garagetotalsqft'] = m



m = round(np.median(xtest['heatingorsystemtypeid'].dropna()))

xtest.loc[xtest['heatingorsystemtypeid'].isnull(),'heatingorsystemtypeid'] = m



m = round(np.median(xtest['lotsizesquarefeet'].dropna()))

xtest.loc[xtest['lotsizesquarefeet'].isnull(),'lotsizesquarefeet'] = m



m = round(np.median(xtest['airconditioningtypeid'].dropna()))

xtest.loc[xtest['airconditioningtypeid'].isnull(), 'airconditioningtypeid'] = m



xtest.loc[(xtest['structuretaxvaluedollarcnt']== xtest['taxvaluedollarcnt'])&(xtest['landtaxvaluedollarcnt'].isnull() == True), ['landtaxvaluedollarcnt']] = 0.0

xtest.loc[(xtest['landtaxvaluedollarcnt']== xtest['taxvaluedollarcnt'])&(xtest['structuretaxvaluedollarcnt'].isnull() == True), ['structuretaxvaluedollarcnt']] = 0.0

xtest.loc[(xtest['taxvaluedollarcnt'].isnull()==False) & (xtest['taxamount'].isnull()==True),['taxamount']] = xtest.loc[(xtest['taxvaluedollarcnt'].isnull()==False) & (xtest['taxamount'].isnull()==True),['taxvaluedollarcnt']] / 40

m = round(np.median(xtest['taxamount'].dropna()))

xtest.loc[xtest['taxamount'].isnull(), 'taxamount'] = m

rec = (xtest.loc[(xtest['structuretaxvaluedollarcnt'].isnull() == True) & (xtest['landtaxvaluedollarcnt'].isnull()==True) & (xtest['taxvaluedollarcnt'].isnull()==True),['taxamount']] * 40 )

xtest.loc[(xtest['structuretaxvaluedollarcnt'].isnull() == True) & (xtest['landtaxvaluedollarcnt'].isnull()==True) & (xtest['taxvaluedollarcnt'].isnull()==True),['taxvaluedollarcnt']] = rec.values

print(rec.values)

rec = (xtest.loc[(xtest['structuretaxvaluedollarcnt'].isnull() == True) & 

          (xtest['landtaxvaluedollarcnt'].isnull()==True),

          ['taxvaluedollarcnt']]) / 2

print (rec)

xtest.loc[(xtest['structuretaxvaluedollarcnt'].isnull() == True) & 

          (xtest['landtaxvaluedollarcnt'].isnull()==True),

          ['structuretaxvaluedollarcnt','landtaxvaluedollarcnt',]] = rec.values







m = 2.5

m = round(np.median(xtest['fullbathcnt'].dropna()))

xtest.loc[xtest['fullbathcnt'].isnull(), 'fullbathcnt'] = m

print (m)



m = round(np.median(xtest['threequarterbathnbr'].dropna()))

xtest.loc[xtest['threequarterbathnbr'].isnull(), 'threequarterbathnbr'] = m

print (m)



m = round(np.median(xtest['lotsizesquarefeet'].dropna()))

xtest.loc[xtest['lotsizesquarefeet'].isnull(), 'lotsizesquarefeet'] = m

print(m)
dtest = xgb.DMatrix(xtest)

ypred = model.predict(dtest)

print (ypred)

pred = []

for i,predict in enumerate(ypred):

    pred.append(str(round(predict,4)))

pred=np.array(pred)

print (pred)

output = pd.DataFrame({

    'ParcelId' : sample['ParcelId'].astype(np.int32),

    '201610' : pred,

    '201611' : pred,

    '201612' : pred,

    '201710' : pred,

    '201711' : pred,

    '201712' : pred

})

#move parceldid to first column

cols = output.columns.tolist()

print (cols)

cols = cols[-1:] + cols[:-1]

output = output[cols]

print (output)

from datetime import datetime

print (datetime.now().strftime('%y%m%d_%H%M'))

output.to_csv('zillow{}.csv'.format(datetime.now().strftime('%y%m%d_%H%M')), index = False)