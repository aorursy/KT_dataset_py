# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings('ignore')
train_data_1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data_1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_data = train_data_1.copy()

test_data = test_data_1.copy()
train_data.head()
train_data.describe()
train_data.shape
train_data.columns
train_data.rename(str.lower, axis='columns',inplace=True)

test_data.rename(str.lower, axis='columns',inplace=True)
train_data.columns
pd.set_option('display.max_columns',None)

train_data.head()
train_data.shape
import missingno as mnso

mnso.bar(train_data,labels=True,sort='ascending') #data in each column
# more then 40% of data is missing

train_data.drop(['alley','fireplacequ','poolqc','fence', 'miscfeature'],axis=1,inplace=True)

test_data.drop(['alley','fireplacequ','poolqc','fence', 'miscfeature'],axis=1,inplace=True)
train_data.shape
sns.kdeplot(train_data['saleprice'].values).set_title("Distribution of saleprice")

plt.xlabel('saleprice')

plt.show()
row_ind = train_data[train_data['saleprice'] > 450000].index.tolist()

row_ind
train_data.drop(row_ind,inplace=True)
train_data.columns
from datetime import date 
today = date.today()
current_year=today.year

current_year
# extracting the house age

train_data['building_age']=current_year - train_data.yearbuilt

test_data['building_age']=current_year - test_data.yearbuilt
# extracting the garagr age

train_data['garage_age']=current_year - train_data.garageyrblt

test_data['garage_age']=current_year - test_data.garageyrblt
# creating the categorial column of wheather the house was remodelled or not

remodelled=[]
for i in list(range(len(train_data)+14)):

    try:

        if train_data.yearremodadd[i] > train_data.yearbuilt[i]:

            remodelled.append(1)

        else:

            remodelled.append(0)

    except:

        pass
train_data.insert(loc=77,column='remodelled', value=remodelled)
remodelled=[]



for i in list(range(len(test_data))):

    if test_data.yearremodadd[i] > test_data.yearbuilt[i]:

        remodelled.append(1)

    else:

        remodelled.append(0)



test_data.insert(loc=77,column='remodelled', value=remodelled)
train_data.head(5)
train_data.drop(['lotfrontage', 'lotarea','lotshape','lotconfig','condition2','exterior1st','exterior2nd'],axis=1,inplace=True)

test_data.drop(['lotfrontage', 'lotarea','lotshape','lotconfig','condition2','exterior1st','exterior2nd'],axis=1,inplace=True)
train_data['sold_date']=pd.DataFrame(train_data[['mosold','yrsold']].apply(lambda x : '{}-{}'.format(x[0],x[1]), axis=1))

test_data['sold_date']=pd.DataFrame(test_data[['mosold','yrsold']].apply(lambda x : '{}-{}'.format(x[0],x[1]), axis=1))
train_data.drop(['yearbuilt', 'yearremodadd','mosold','yrsold','garageyrblt'],axis=1,inplace=True)

test_data.drop(['yearbuilt', 'yearremodadd','mosold','yrsold','garageyrblt'],axis=1,inplace=True)
train_data.head(5)
train_data.drop('id',axis=1,inplace=True)

test_data.drop('id',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
Y = train_data['saleprice']

X = train_data.copy().drop('saleprice',axis=1)
trainx,valx,trainy,valy=train_test_split(X,Y,test_size=0.30,random_state=123)
trainx.columns
import datetime
trainx['sold_date']=pd.to_datetime(trainx['sold_date'],format='%m-%Y')
test_data['sold_date']=pd.to_datetime(test_data['sold_date'],format='%m-%Y')
trainx.drop(['garage_age','garagearea'],axis=1,inplace=True)
test_data.drop(['garage_age','garagearea'],axis=1,inplace=True)
cat_cols=['mssubclass', 'mszoning', 'street','landcontour', 'utilities','landslope','neighborhood', 'condition1','bldgtype',

          'housestyle','overallqual', 'overallcond','roofstyle','roofmatl','masvnrtype','exterqual','extercond','foundation',

          'bsmtqual', 'bsmtcond','bsmtexposure', 'bsmtfintype1', 'bsmtfintype2','heating', 'heatingqc','centralair', 'electrical',

          'kitchenqual','functional','garagetype','garagefinish','garagequal', 'garagecond', 'paveddrive','saletype',

          'salecondition','remodelled']

num_cols=['1stflrsf', '2ndflrsf', '3ssnporch', 'bedroomabvgr', 'bsmtfinsf1','bsmtfinsf2', 'bsmtfullbath', 'bsmthalfbath', 

          'bsmtunfsf','building_age', 'enclosedporch', 'fireplaces', 'fullbath','garagecars','grlivarea', 'halfbath','kitchenabvgr', 'lowqualfinsf', 'masvnrarea', 'miscval', 'openporchsf','poolarea', 

          'screenporch', 'totalbsmtsf', 'totrmsabvgrd','wooddecksf']
test_cat_cols=['mssubclass', 'mszoning', 'street','landcontour', 'utilities','landslope','neighborhood', 'condition1','bldgtype',

          'housestyle','overallqual', 'overallcond','roofstyle','roofmatl','masvnrtype','exterqual','extercond','foundation',

          'bsmtqual', 'bsmtcond','bsmtexposure', 'bsmtfintype1', 'bsmtfintype2','heating', 'heatingqc','centralair', 'electrical',

          'kitchenqual','functional','garagetype','garagefinish','garagequal', 'garagecond', 'paveddrive','saletype',

          'salecondition','remodelled']

test_num_cols=['1stflrsf', '2ndflrsf', '3ssnporch', 'bedroomabvgr', 'bsmtfinsf1','bsmtfinsf2', 'bsmtfullbath', 'bsmthalfbath', 

          'bsmtunfsf','building_age', 'enclosedporch', 'fireplaces', 'fullbath','garagecars','grlivarea', 'halfbath','kitchenabvgr', 'lowqualfinsf', 'masvnrarea', 'miscval', 'openporchsf','poolarea', 

          'screenporch', 'totalbsmtsf', 'totrmsabvgrd','wooddecksf']
trainx[cat_cols]=trainx[cat_cols].apply(lambda x : x.astype('category'))

trainx[num_cols]=trainx[num_cols].apply(lambda x : x.astype('float64'))
test_data[test_cat_cols]=test_data[test_cat_cols].apply(lambda x : x.astype('category'))

test_data[test_num_cols]=test_data[test_num_cols].apply(lambda x : x.astype('float64'))
cormat = trainx.corr()

f , ax = plt.subplots(figsize=(30,30))

sns.heatmap(cormat,ax=ax,cmap="YlGnBu" ,linewidths=0.5,annot=True)
cat_data=trainx.loc[:,cat_cols]

num_data=trainx.loc[:,num_cols]
test_cat_data=test_data.loc[:,test_cat_cols]

test_num_data=test_data.loc[:,test_num_cols]
from sklearn.impute import SimpleImputer
imp_cat = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp_cat.fit(cat_data)

cat_data=pd.DataFrame(imp_cat.transform(cat_data),columns=cat_cols)

imp_num = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_num.fit(num_data)

num_data=pd.DataFrame(imp_num.transform(num_data),columns=num_cols)
from sklearn.preprocessing import StandardScaler
#Coverting train data int Z-Scores

standardizer = StandardScaler()

standardizer.fit(num_data)

num_data = pd.DataFrame(standardizer.transform(num_data),index=num_data.index,columns=num_cols)

trainx = pd.merge(num_data,cat_data,left_on=num_data.index,right_on=cat_data.index,how='inner',left_index=True)

#Dropping extra column which has come after the joins

trainx.drop('key_0',axis=1,inplace=True)
trainx=pd.get_dummies(trainx,columns=['remodelled','centralair'],drop_first=True)
import category_encoders as ce
encoder = ce.BinaryEncoder(cols=['mssubclass','neighborhood','saletype', 'salecondition','condition1','masvnrtype','heating',

                             'electrical','garagetype'])

temp1=encoder.fit(trainx)

trainx=temp1.transform(trainx)
trainx.head(1)
mszoning_dict={'A':1,'C (all)':2,'FV':3,'I':4,'RH':5,'RL':6,'RP':7,'RM':8}

street_dict={'Grvl':1,'Pave':2}

landcontour_dict={'Low':1,'HLS':2,'Bnk':3,'Lvl':4}

utilities_dict={'NoSeWa':1,'AllPub':2}

garagefinish_dict={'nan':1,'Unf':2,'RFn':3,'Fin':4}

garagequal_dict={'nan':1,'Po':2,'Fa':3,'TA':4,'Gd':5,'Ex':6}

garagecond_dict={'nan':1,'Po':2,'Fa':3,'TA':4,'Gd':5,'Ex':6}

paveddrive_dict={'N':1,'P':2,'Y':3}

landslope_dict={'Sev':1,'Mod':2,'Gtl':3}

bldgtype_dict={'Twnhs':1,'TwnhsE':2,'Duplex':3,'2fmCon':4,'1Fam':5}

housestyle_dict={'SLvl':1,'SFoyer':2,'2.5Unf':3,'2.5Fin':4,'2Story':5,'1.5Unf':6,'1.5Fin':7,'1Story':8}

kitchenqual_dict={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

functional_dict={'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8}

roofstyle_dict={'Shed':1,'Mansard':2,'Hip':3,'Gambrel':4,'Gable':5,'Flat':6}

roofmatl_dict={'WdShngl':1,'WdShake':2,'Tar&Grv':3,'Roll':4,'Metal':5,'Membran':6,'CompShg':7,'ClyTile':8}

exterqual_dict={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

extercond_dict={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

foundation_dict={'Wood':1,'Stone':2,'Slab':3,'PConc':4,'CBlock':5,'BrkTil':6}

bsmtqual_dict={'nan':1,'Po':2,'Fa':3,'TA':4,'Gd':5,'Ex':6}

bsmtcond_dict={'nan':1,'Po':2,'Fa':3,'TA':4,'Gd':5,'Ex':6}

bsmtexposure_dict={'nan':1,'No':2,'Mn':3,'Av':4,'Gd':5}

bsmtfintype1_dict={'nan':1,'Unf':2,'LwQ':3,'Rec':4,'BLQ':5,'ALQ':6,'GLQ':7}

bsmtfintype2_dict={'nan':1,'Unf':2,'LwQ':3,'Rec':4,'BLQ':5,'ALQ':6,'GLQ':7}

heatingqc_dict={'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}

overallqual_dict={1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10}

overallcond_dict={1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10}
trainx['mszoning'] = trainx.mszoning.map(mszoning_dict)

trainx['street'] = trainx.street.map(street_dict)

trainx['landcontour'] = trainx.landcontour.map(landcontour_dict)

trainx['utilities'] = trainx.utilities.map(utilities_dict)

trainx['garagefinish'] = trainx.garagefinish.map(garagefinish_dict)

trainx['garagequal'] = trainx.garagequal.map(garagequal_dict)

trainx['garagecond'] = trainx.garagecond.map(garagecond_dict)

trainx['paveddrive'] = trainx.paveddrive.map(paveddrive_dict)

trainx['landslope'] = trainx.landslope.map(landslope_dict)

trainx['bldgtype'] = trainx.bldgtype.map(bldgtype_dict)

trainx['housestyle'] = trainx.housestyle.map(housestyle_dict)

trainx['kitchenqual'] = trainx.kitchenqual.map(kitchenqual_dict)

trainx['functional'] = trainx.functional.map(functional_dict)

trainx['roofstyle'] = trainx.roofstyle.map(roofstyle_dict)

trainx['roofmatl'] = trainx.roofmatl.map(roofmatl_dict)

trainx['exterqual'] = trainx.exterqual.map(exterqual_dict)

trainx['extercond'] = trainx.extercond.map(extercond_dict)

trainx['foundation'] = trainx.foundation.map(foundation_dict)

trainx['bsmtqual'] = trainx.bsmtqual.map(bsmtqual_dict)

trainx['bsmtcond'] = trainx.bsmtcond.map(bsmtcond_dict)

trainx['bsmtexposure'] = trainx.bsmtexposure.map(bsmtexposure_dict)

trainx['bsmtfintype1'] = trainx.bsmtfintype1.map(bsmtfintype1_dict)

trainx['bsmtfintype2'] = trainx.bsmtfintype2.map(bsmtfintype2_dict)

trainx['heatingqc'] = trainx.heatingqc.map(heatingqc_dict)

trainx['overallqual'] = trainx.overallqual.map(overallqual_dict)

trainx['overallcond'] = trainx.overallcond.map(overallcond_dict)
trainx.shape
valx['sold_date']=pd.to_datetime(valx['sold_date'],format='%m-%Y')
valx.drop(['garage_age','garagearea'],axis=1,inplace=True)
val_cat_cols=['mssubclass', 'mszoning', 'street','landcontour', 'utilities','landslope','neighborhood', 'condition1','bldgtype',

          'housestyle','overallqual', 'overallcond','roofstyle','roofmatl','masvnrtype','exterqual','extercond','foundation',

          'bsmtqual', 'bsmtcond','bsmtexposure', 'bsmtfintype1', 'bsmtfintype2','heating', 'heatingqc','centralair', 'electrical',

          'kitchenqual','functional','garagetype','garagefinish','garagequal', 'garagecond', 'paveddrive','saletype',

          'salecondition','remodelled']

val_num_cols=['1stflrsf', '2ndflrsf', '3ssnporch', 'bedroomabvgr', 'bsmtfinsf1','bsmtfinsf2', 'bsmtfullbath', 'bsmthalfbath', 

          'bsmtunfsf','building_age', 'enclosedporch', 'fireplaces', 'fullbath','garagecars','grlivarea', 'halfbath','kitchenabvgr', 'lowqualfinsf', 'masvnrarea', 'miscval', 'openporchsf','poolarea', 

          'screenporch', 'totalbsmtsf', 'totrmsabvgrd','wooddecksf']

valx[val_cat_cols]=valx[val_cat_cols].apply(lambda x : x.astype('category'))

valx[val_num_cols]=valx[val_num_cols].apply(lambda x : x.astype('float64'))
val_cat_data=valx.loc[:,val_cat_cols]

val_num_data=valx.loc[:,val_num_cols]
val_cat_data=pd.DataFrame(imp_cat.transform(val_cat_data),columns=val_cat_cols)

val_num_data=pd.DataFrame(imp_num.transform(val_num_data),columns=val_num_cols)
val_num_data = pd.DataFrame(standardizer.transform(val_num_data),index=val_num_data.index,columns=val_num_cols)
valx = pd.merge(val_num_data,val_cat_data,left_on=val_num_data.index,right_on=val_cat_data.index,how='inner',left_index=True)

valx.drop('key_0',axis=1,inplace=True)
valx=pd.get_dummies(valx,columns=['remodelled','centralair'],drop_first=True)
valx=temp1.transform(valx)
valx['mszoning'] = valx.mszoning.map(mszoning_dict)

valx['street'] = valx.street.map(street_dict)

valx['landcontour'] = valx.landcontour.map(landcontour_dict)

valx['utilities'] = valx.utilities.map(utilities_dict)

valx['garagefinish'] = valx.garagefinish.map(garagefinish_dict)

valx['garagequal'] = valx.garagequal.map(garagequal_dict)

valx['garagecond'] = valx.garagecond.map(garagecond_dict)

valx['paveddrive'] = valx.paveddrive.map(paveddrive_dict)

valx['landslope'] = valx.landslope.map(landslope_dict)

valx['bldgtype'] = valx.bldgtype.map(bldgtype_dict)

valx['housestyle'] = valx.housestyle.map(housestyle_dict)

valx['kitchenqual'] = valx.kitchenqual.map(kitchenqual_dict)

valx['functional'] = valx.functional.map(functional_dict)

valx['roofstyle'] = valx.roofstyle.map(roofstyle_dict)

valx['roofmatl'] = valx.roofmatl.map(roofmatl_dict)

valx['exterqual'] = valx.exterqual.map(exterqual_dict)

valx['extercond'] = valx.extercond.map(extercond_dict)

valx['foundation'] = valx.foundation.map(foundation_dict)

valx['bsmtqual'] = valx.bsmtqual.map(bsmtqual_dict)

valx['bsmtcond'] = valx.bsmtcond.map(bsmtcond_dict)

valx['bsmtexposure'] = valx.bsmtexposure.map(bsmtexposure_dict)

valx['bsmtfintype1'] = valx.bsmtfintype1.map(bsmtfintype1_dict)

valx['bsmtfintype2'] = valx.bsmtfintype2.map(bsmtfintype2_dict)

valx['heatingqc'] = valx.heatingqc.map(heatingqc_dict)

valx['overallqual'] = valx.overallqual.map(overallqual_dict)

valx['overallcond'] = valx.overallcond.map(overallcond_dict)
valx.shape
trainx.columns
trainx.shape

test_data.shape
test_cat_data=pd.DataFrame(imp_cat.transform(test_cat_data),columns=test_cat_cols)

test_num_data=pd.DataFrame(imp_num.transform(test_num_data),columns=test_num_cols)
test_num_data = pd.DataFrame(standardizer.transform(test_num_data),index=test_num_data.index,columns=test_num_cols)
test_data = pd.merge(test_num_data,test_cat_data,left_on=test_num_data.index,right_on=test_cat_data.index,how='inner',left_index=True)

test_data.drop('key_0',axis=1,inplace=True)
test_data.columns
test_data.head(1)
test_data=pd.get_dummies(test_data,columns=['remodelled','centralair'],drop_first=True)
test_data.columns
test_data.head(2)
test_data=temp1.transform(test_data)
test_data['mszoning'] = test_data.mszoning.map(mszoning_dict)

test_data['street'] = test_data.street.map(street_dict)

test_data['landcontour'] = test_data.landcontour.map(landcontour_dict)

test_data['utilities'] = test_data.utilities.map(utilities_dict)

test_data['garagefinish'] = test_data.garagefinish.map(garagefinish_dict)

test_data['garagequal'] = test_data.garagequal.map(garagequal_dict)

test_data['garagecond'] = test_data.garagecond.map(garagecond_dict)

test_data['paveddrive'] = test_data.paveddrive.map(paveddrive_dict)

test_data['landslope'] = test_data.landslope.map(landslope_dict)

test_data['bldgtype'] = test_data.bldgtype.map(bldgtype_dict)

test_data['housestyle'] = test_data.housestyle.map(housestyle_dict)

test_data['kitchenqual'] = test_data.kitchenqual.map(kitchenqual_dict)

test_data['functional'] = test_data.functional.map(functional_dict)

test_data['roofstyle'] = test_data.roofstyle.map(roofstyle_dict)

test_data['roofmatl'] = test_data.roofmatl.map(roofmatl_dict)

test_data['exterqual'] = test_data.exterqual.map(exterqual_dict)

test_data['extercond'] = test_data.extercond.map(extercond_dict)

test_data['foundation'] = test_data.foundation.map(foundation_dict)

test_data['bsmtqual'] = test_data.bsmtqual.map(bsmtqual_dict)

test_data['bsmtcond'] = test_data.bsmtcond.map(bsmtcond_dict)

test_data['bsmtexposure'] = test_data.bsmtexposure.map(bsmtexposure_dict)

test_data['bsmtfintype1'] = test_data.bsmtfintype1.map(bsmtfintype1_dict)

test_data['bsmtfintype2'] = test_data.bsmtfintype2.map(bsmtfintype2_dict)

test_data['heatingqc'] = test_data.heatingqc.map(heatingqc_dict)

test_data['overallqual'] = test_data.overallqual.map(overallqual_dict)

test_data['overallcond'] = test_data.overallcond.map(overallcond_dict)
from sklearn import metrics

def rmse(train_actual,train_prediction,test_actual,test_prediction):

    print('train')

    print('rmse: ',np.sqrt(metrics.mean_squared_error(train_actual, train_prediction)))

    print('Validation')

    print('rmse: ',np.sqrt(metrics.mean_squared_error(test_actual, test_prediction)))
from xgboost import XGBRegressor
%%time

xgb = XGBRegressor()

xgb.fit(trainx,trainy)
predictions_train_xgb = xgb.predict(trainx)

predictions_val_xgb = xgb.predict(valx)
rmse(trainy,predictions_train_xgb,valy,predictions_val_xgb)
predictions_test_xgb2 = xgb.predict(test_data)
# GridSearch
param_grid_xgb = {"criterion": ["mse", "mae"],

              "min_samples_split": [10, 20, 40],

              "max_depth": [2, 6, 8],

              "min_samples_leaf": [20, 40, 100],

              "max_leaf_nodes": [5, 20, 100],

              }
%%time

grid_xgb = GridSearchCV(xgb,param_grid=param_grid_xgb,cv=10,n_jobs=-1)

grid_xgb.fit(trainx,trainy)
print(grid_xgb.best_estimator_)
predictions_train_grid_xgb = grid_xgb.predict(trainx)

predictions_val_grid_xgb = grid_xgb.predict(valx)
rmse(trainy,predictions_train_grid_xgb,valy,predictions_val_grid_xgb)