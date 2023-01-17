import pandas as pd
import matplotlib as plt
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from datetime import date
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost as xgb
import time
import ast

%matplotlib inline
df_train = pd.read_csv('../input/autos/autos_training_final.csv', sep=',')
df_test = pd.read_csv('../input/autos/autos_testing_final.csv', sep=',')
df = pd.concat([df_train, df_test], ignore_index=True)
display(df.describe())
display(df.info())
df.head()
# Ismeretlen értékek
df.isnull().sum()
# seller
df['seller'].value_counts()
# offerType
df['offerType'].value_counts()
# Year of registration
df['yearOfRegistration'] = df['yearOfRegistration'].astype(int)
chart = df.loc[df.yearOfRegistration >= 1500, 'yearOfRegistration']
mm = chart.max() - chart.min()
chart.hist(bins=mm)
# Month of registration
df['monthOfRegistration'] = df['monthOfRegistration'].astype(int)
df.monthOfRegistration.hist(bins=12)
#df[(df.yearOfRegistration == 1000)].values
# Number of pictures
df.nrOfPictures.value_counts()
# Postal Code
df.postalCode.hist(bins=10000)
# Vehicle type
df.vehicleType.fillna('NaN', inplace=True)
df.vehicleType.value_counts().plot(kind='bar')
# FuelType
df.fuelType.fillna('NaN', inplace=True)
df.fuelType.value_counts().plot(kind='bar')
# Gearbox
df.gearbox.value_counts()
# Kilometer
df.kilometer.hist(bins=50)
# Model
df.model.fillna('NaN', inplace=True)
df.model.value_counts().plot(kind='bar')
# notRepairedDamage
df.notRepairedDamage.fillna('NaN', inplace=True)
df.notRepairedDamage.value_counts().plot(kind='bar')
# powerPS
df.powerPS = pd.to_numeric(df.powerPS, downcast='integer')
df.powerPS[df.powerPS <= 1000].hist(bins=100)
print('0: \t', df.powerPS[df.powerPS <= 10].shape[0], 'db')
print('0-1000:\t', df.powerPS[(df.powerPS > 0) & (df.powerPS < 1000)].shape[0], 'db')
print('1000-:\t', df.powerPS[(df.powerPS >= 1000)].shape[0], 'db')
df['lastSeen'] = df.lastSeen.astype('datetime64[ns]').dt.date
df['daysElapsed'] = (df.lastSeen.astype('datetime64[ns]').dt.date - df.dateCreated.astype('datetime64[ns]').dt.date).dt.days
df['isAutomatik'] = df['gearbox'].apply(lambda x: 1 if x=='automatik' else 0)
df['isManuell'] = df['gearbox'].apply(lambda x: 1 if x=='manuell' else 0)
df['isGearboxUnknown'] = df['gearbox'].apply(lambda x: 1 if x=='NaN' else 0)
def createTurboColumn(dataF):
    return dataF.name.str.contains('Turbo', case=False) |\
           dataF.name.str.contains('tsi', case=False) |\
           dataF.name.str.contains('tdi', case=False) |\
           dataF.name.str.contains('tfsi', case=False)

def createSearchOrSwapColumn(dataF):
    return dataF.name.str.contains('such', case=False) |\
           dataF.name.str.contains('tausch', case=False)

def createKlimaColumn(dataF):
    return dataF.name.str.contains('klima', case=False)

def createDPFColumn(dataF):
    return dataF.name.str.contains('dpf', case=False)

def createDSGColumn(dataF):
    return dataF.name.str.contains('dsg', case=False)

def createGTColumn(dataF):
    return dataF.name.str.contains('gti', case=False) |\
           dataF.name.str.contains('gtd', case=False)


df['hasTurbo'] = createTurboColumn(df).map({True:1, False:0})
df['isSearchOrSwap'] = createSearchOrSwapColumn(df).map({True:1, False:0})
df['hasKlima'] = createKlimaColumn(df).map({True:1, False:0})
df['hasDPF'] = createDPFColumn(df).map({True:1, False:0})
df['hasDSG'] = createDSGColumn(df).map({True:1, False:0})
df['hasGTIorGTD'] = createGTColumn(df).map({True:1, False:0})
df['nameLength'] = df.name.apply(lambda x: len(x))
df['nameWordsCount'] = df.name.apply(lambda x: len(x.split('_')))
def fillModelMissingGr(group):
    modelList = list(df.loc[df.brand == group.iloc[0]['brand'], 'model'].unique())
    modelList = sorted(modelList, key=lambda x: (1 / len(x), x.lower()))
    for mod in modelList:
        group.loc[(group['name'].str.contains(mod, case=False)) & (group['model'] == 'NaN'), 'model'] = mod
    return group['model']
# Model ismeretlen értékeinek kitalálása
df.loc[(df.model == 'NaN') & (df.brand != 'sonstige_autos'), 'model'] = \
        df.loc[(df.model == 'NaN') & (df.brand != 'sonstige_autos')]\
    .groupby('brand').apply(fillModelMissingGr).reset_index(level=0)['model']
# Mi számít nagynak, és mi kicsinek? Ezt most konstansokkal döntöm el
const_high_power = 1000
const_low_power = 10
df['powerHIGH'] = (df['powerPS'] >= const_high_power).map({True:1, False:0})
df['powerLOW'] = (df['powerPS'] <= const_low_power).map({True:1, False:0})
def fillPowerOutliers(row):
    # először megnézem, hogy a hármas szerepel-e az indexek között
    if (row.brand, row.model, row.vehicleType) in powerTable.index:
        return powerTable.loc[row.brand, row.model, row.vehicleType]['mean']
    else:
        if (row.brand, row.model) in powerTable.index:
            return powerTable.loc[row.brand, row.model]['mean'].mean()
        else:
            return powerTable.loc[row.brand]['mean'].mean()
# PowerPS átlag táblázat
powerTable = df[(df.powerHIGH == 0) & (df.powerLOW == 0)].groupby(['brand', 'model', 'vehicleType'])['powerPS'].agg(['mean', 'std'])
# behelyettesítjük a kilógó értékek helyére
df.loc[(df.powerHIGH == 1) | (df.powerLOW == 1),'powerPS'] = df.loc[(df.powerHIGH == 1) | (df.powerLOW == 1)].apply(fillPowerOutliers, axis=1) 
df['carNameYear'] = df['brand'] + df['model'] + df['yearOfRegistration'].astype(str)
df['carName'] = df['brand'] + df['model']
df.loc[(df.fuelType == 'NaN') & ((df.name.str.contains('dpf', case=False)) | \
       (df.name.str.contains('tdi', case=False))), 'fuelType'] = 'diesel'
df['isDamageYes'] = df['notRepairedDamage'].apply(lambda x: 1 if x=='ja' else 0)
df['isDamageNo'] = df['notRepairedDamage'].apply(lambda x: 1 if x=='nein' else 0)
df['isDamageNaN'] = df['notRepairedDamage'].apply(lambda x: 1 if x=='NaN' else 0)
fuelDict = dict(zip(df.fuelType.value_counts().index, df.fuelType.value_counts().values))
vehicleDict = dict(zip(df.vehicleType.value_counts().index, df.vehicleType.value_counts().values))
modelDict = dict(zip(df.model.value_counts().index, df.model.value_counts().values))
brandDict = dict(zip(df.brand.value_counts().index, df.brand.value_counts().values))
carNameDict = dict(zip(df.carName.value_counts().index, df.carName.value_counts().values))
df_train = pd.read_csv('../input/autos/autos_training_final.csv', sep=',')
# Konvertálás, ha kell
df_train['yearOfRegistration'] = df_train.yearOfRegistration.astype(int)

# Ismeretlenek kezelése:
df_train.fuelType.fillna('NaN', inplace=True)
df_train.model.fillna('NaN', inplace=True)
df_train.vehicleType.fillna('NaN', inplace=True)
df_train.gearbox.fillna('NaN', inplace=True)
df_train.notRepairedDamage.fillna('NaN', inplace=True)

# Outlierek kezelése
df_train.loc[df_train['monthOfRegistration'] == 0, 'monthOfRegistration'] = -9999
df_train.loc[df_train['yearOfRegistration'] == 1000, 'yearOfRegistration'] = -9999

# Létrehozott featurek és adatmanipuláció elvégzése
df_train['isAutomatik'] = df_train['gearbox'].apply(lambda x: 1 if x=='automatik' else 0)
df_train['isManuell'] = df_train['gearbox'].apply(lambda x: 1 if x=='manuell' else 0)
df_train['isGearboxUnknown'] = df_train['gearbox'].apply(lambda x: 1 if x=='NaN' else 0)

df_train['isDamageYes'] = df_train['notRepairedDamage'].apply(lambda x: 1 if x=='ja' else 0)
df_train['isDamageNo'] = df_train['notRepairedDamage'].apply(lambda x: 1 if x=='nein' else 0)
df_train['isDamageNaN'] = df_train['notRepairedDamage'].apply(lambda x: 1 if x=='NaN' else 0)

df_train['powerHIGH'] = (df_train['powerPS'] >= const_high_power).map({True:1, False:0})
df_train['powerLOW'] = (df_train['powerPS'] <= const_low_power).map({True:1, False:0})

df_train.loc[(df_train.powerHIGH == 1) | (df_train.powerLOW == 1),'powerPS'] = \
            df_train.loc[(df_train.powerHIGH == 1) | (df_train.powerLOW == 1)].apply(fillPowerOutliers, axis=1)

df_train['carNameYear'] = df_train['brand'] + df_train['model'] + df_train['yearOfRegistration'].astype(str)

df_train.loc[(df_train.model == 'NaN') & (df_train.brand != 'sonstige_autos'), 'model'] = \
            df_train.loc[(df_train.model == 'NaN') & (df_train.brand != 'sonstige_autos')] \
            .groupby('brand').apply(fillModelMissingGr).reset_index(level=0)['model']

df_train['carName'] = df_train['brand'] + df_train['model']

df_train.loc[(df_train.fuelType == 'NaN') & \
             ((df_train.name.str.contains('dpf', case=False)) | \
              (df_train.name.str.contains('tdi', case=False))), 'fuelType'] = 'diesel'

df_train['hasTurbo'] = createTurboColumn(df_train).map({True:1, False:0})
df_train['isSearchOrSwap'] = createSearchOrSwapColumn(df_train).map({True:1, False:0})
df_train['hasKlima'] = createKlimaColumn(df_train).map({True:1, False:0})
df_train['hasDPF'] = createDPFColumn(df_train).map({True:1, False:0})
df_train['hasDSG'] = createDSGColumn(df_train).map({True:1, False:0})
df_train['hasGTIorGTD'] = createGTColumn(df_train).map({True:1, False:0})

df_train['nameLength'] = df_train.name.apply(lambda x: len(x))
df_train['nameWordsCount'] = df_train.name.apply(lambda x: len(x.split('_')))

df_train['lastSeen'] = df_train.lastSeen.astype('datetime64[ns]').dt.date
df_train['daysElapsed'] = (df_train.lastSeen.astype('datetime64[ns]').dt.date - \
                           df_train.dateCreated.astype('datetime64[ns]').dt.date).dt.days
df_train['brand'] = df_train.brand.apply(lambda x: brandDict.get(x))
df_train['model'] = df_train.model.apply(lambda x: modelDict.get(x))
df_train['vehicleType'] = df_train.vehicleType.apply(lambda x: vehicleDict.get(x))
df_train['fuelType'] = df_train.fuelType.apply(lambda x: fuelDict.get(x))
df_train['carName'] = df_train.carName.apply(lambda x: carNameDict.get(x))

lcoder_gearbox = LabelEncoder()
df_train.gearbox = lcoder_gearbox.fit_transform(df_train.gearbox)

lcoder_carNameYear = LabelEncoder()
df_train.carNameYear = lcoder_carNameYear.fit_transform(df_train.carNameYear)

lcoder_notRepairedDamage = LabelEncoder()
df_train.notRepairedDamage = lcoder_notRepairedDamage.fit_transform(df_train.notRepairedDamage)

lcoder_lastSeen = LabelEncoder()
df_train.lastSeen = lcoder_lastSeen.fit_transform(df_train.lastSeen)
features = ['brand','model','carName','vehicleType', 'fuelType', 'kilometer',\
            'yearOfRegistration','hasTurbo', 'isSearchOrSwap',  'powerLOW',\
            'powerPS', 'isAutomatik', 'isGearboxUnknown' , 'isManuell',  'hasKlima',\
            'nameWordsCount', 'monthOfRegistration','nameLength', 'hasDPF', \
            'lastSeen', 'daysElapsed', 'carNameYear','powerHIGH', 'hasDSG', 'hasGTIorGTD',\
            'isDamageYes', 'isDamageNaN', 'isDamageNo', 'postalCode']

target = 'label'
X = df_train[features].copy()
Y = df_train[target].copy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=19)
def evaluateModel(model):
    model.fit(X_train, Y_train)
    result_prob=model.predict_proba(X_test)
    results=model.predict(X_test)
    acc = accuracy_score(Y_test, results)
    dfresults = pd.DataFrame(result_prob,columns=["Prob_0","Prob_1"])
    auc = auc_score(Y_test, dfresults["Prob_1"])
    logl = log_loss(Y_test, result_prob)
    print("Acc="+str(acc)+"\tAUC="+str(auc)+'\tLogL='+str(logl))
    
def plotFeatureImportance(model):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    featureDf = pd.DataFrame(data = feature_importance, index=features, columns=['imp'])
    featureDf.sort_values('imp').plot(kind='bar')
model2 = XGBClassifier(learning_rate=0.06, n_estimators=800, max_depth=11, min_child_weight=1, gamma=0.8,\
                       subsample=0.8, colsample_bytree=0.8, silent=False, n_jobs=16, random_state=10)
evaluateModel(model2)
plotFeatureImportance(model2)
df_test = pd.read_csv('../input/autos/autos_testing_final.csv', sep=',')

# Konvertálás, ha kell
df_test['yearOfRegistration'] = df_test.yearOfRegistration.astype(int)

# Ismeretlenek kezelése:
df_test.fuelType.fillna('NaN', inplace=True)
df_test.model.fillna('NaN', inplace=True)
df_test.vehicleType.fillna('NaN', inplace=True)
df_test.gearbox.fillna('NaN', inplace=True)
df_test.notRepairedDamage.fillna('NaN', inplace=True)

# Outlierek kezelése
df_test.loc[df_test['monthOfRegistration'] == 0, 'monthOfRegistration'] = -9999
df_test.loc[df_test['yearOfRegistration'] == 1000, 'yearOfRegistration'] = -9999

# Létrehozott featurek és adatmanipuláció elvégzése
df_test['isAutomatik'] = df_test['gearbox'].apply(lambda x: 1 if x=='automatik' else 0)
df_test['isManuell'] = df_test['gearbox'].apply(lambda x: 1 if x=='manuell' else 0)
df_test['isGearboxUnknown'] = df_test['gearbox'].apply(lambda x: 1 if x=='NaN' else 0)

df_test['isDamageYes'] = df_test['notRepairedDamage'].apply(lambda x: 1 if x=='ja' else 0)
df_test['isDamageNo'] = df_test['notRepairedDamage'].apply(lambda x: 1 if x=='nein' else 0)
df_test['isDamageNaN'] = df_test['notRepairedDamage'].apply(lambda x: 1 if x=='NaN' else 0)

df_test['powerHIGH'] = (df_test['powerPS'] >= const_high_power).map({True:1, False:0})
df_test['powerLOW'] = (df_test['powerPS'] <= const_low_power).map({True:1, False:0})

df_test.loc[(df_test.powerHIGH == 1) | (df_test.powerLOW == 1),'powerPS'] = \
            df_test.loc[(df_test.powerHIGH == 1) | (df_test.powerLOW == 1)].apply(fillPowerOutliers, axis=1)
    
df_test['carNameYear'] = df_test['brand'] + df_test['model'] + df_test['yearOfRegistration'].astype(str)

df_test.loc[(df_test.model == 'NaN') & (df_test.brand != 'sonstige_autos'), 'model'] = \
            df_test.loc[(df_test.model == 'NaN') & (df_test.brand != 'sonstige_autos')]\
            .groupby('brand').apply(fillModelMissingGr).reset_index(level=0)['model']
        
df_test['carName'] = df_test['brand'] + df_test['model']

df_test.loc[(df_test.fuelType == 'NaN') & \
             ((df_test.name.str.contains('dpf', case=False)) | \
              (df_test.name.str.contains('tdi', case=False))), 'fuelType'] = 'diesel'

df_test['hasTurbo'] = createTurboColumn(df_test).map({True:1, False:0})
df_test['isSearchOrSwap'] = createSearchOrSwapColumn(df_test).map({True:1, False:0})
df_test['hasKlima'] = createKlimaColumn(df_test).map({True:1, False:0})
df_test['hasDPF'] = createDPFColumn(df_test).map({True:1, False:0})
df_test['hasDSG'] = createDSGColumn(df_test).map({True:1, False:0})
df_test['hasGTIorGTD'] = createGTColumn(df_test).map({True:1, False:0})
df_test['nameLength'] = df_test.name.apply(lambda x: len(x))
df_test['nameWordsCount'] = df_test.name.apply(lambda x: len(x.split('_')))

df_test['lastSeen'] = df_test.lastSeen.astype('datetime64[ns]').dt.date
df_test['daysElapsed'] = (df_test.lastSeen.astype('datetime64[ns]').dt.date - \
                          df_test.dateCreated.astype('datetime64[ns]').dt.date).dt.days
df_test['brand'] = df_test.brand.apply(lambda x: brandDict.get(x))
df_test['model'] = df_test.model.apply(lambda x: modelDict.get(x))
df_test['vehicleType'] = df_test.vehicleType.apply(lambda x: vehicleDict.get(x))
df_test['fuelType'] = df_test.fuelType.apply(lambda x: fuelDict.get(x))
df_test['carName'] = df_test.carName.apply(lambda x: carNameDict.get(x))

df_test.gearbox = lcoder_gearbox.transform(df_test.gearbox)
df_test.carNameYear = lcoder_carNameYear.transform(df_test.carNameYear)
df_test.notRepairedDamage = lcoder_notRepairedDamage.transform(df_test.notRepairedDamage)
df_test.lastSeen = lcoder_lastSeen.transform(df_test.lastSeen)
# leválasztom azon oszlopokat amit a modell felhasznál.
X_sub = df_test[features].copy()
df_test['label'] = model2.predict_proba(X_sub)[:,1]
# Kiírjuk egy fájlba feltöltésre
df_test[['id', 'label']].to_csv('xgb_06_800_11_1_0.8_encoded.csv', index=False)
