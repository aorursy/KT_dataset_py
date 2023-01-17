###Loading in useful packages

#for linear algebra
import numpy as np

#for data manipulation
import pandas as pd

#for plotting
import matplotlib.pyplot as pp
import seaborn as sns
%matplotlib inline

#For surpressing warnings
import warnings
warnings.filterwarnings('ignore')
#Loading in the data

mini_train = pd.read_pickle("../input/mctrain/train.pkl")
test = pd.read_pickle("../input/mctest/test.pkl")
mini_train = mini_train.reset_index().drop('index',axis=1)
#Let's take a look at the  training data
mini_train.head()
#Lets look at the data types
mini_train.info()
#Let's make a list of the categorical variables here

categorical_vars = list(mini_train.dtypes[mini_train.dtypes== 'category'].index)[1:]

for i in categorical_vars:
  print("Variable Name:",i)
  print(i,"has",len(list(mini_train.loc[:,i].unique())),"unique values",sep=" ")
  print(i,"has",mini_train.loc[:,i].isnull().sum(),"missing values",sep=" ")
  print(" ")
#Dealing with the PuaMode missing values
mini_train['PuaMode']=mini_train['PuaMode'].replace(np.nan,'absent')

#Let's make the same changes to the testing data too
test['PuaMode']=test['PuaMode'].replace(np.nan,'absent')

#Let's take a look at the values
mini_train['PuaMode'].value_counts()

#Dealing with SmartScreen missing values

mini_train['SmartScreen']= mini_train['SmartScreen'].replace([np.nan,'&#x01;','&#x02;','Off','off','absent','OFF','0','of','Deny','ABSENT','ting','ExistsNotSet','Block','BLOCK','Not-there'],'NOT-THERE')
mini_train['SmartScreen']= mini_train['SmartScreen'].replace(['RequireAdmin','requireadmin','RequiredAdmin','Admin','requireAdmin'],'ADMIN')
mini_train['SmartScreen']= mini_train['SmartScreen'].replace(['On','ON','on','Enabled'],'oN')
mini_train['SmartScreen']= mini_train['SmartScreen'].replace(['Prompt','warn','Warn','Promprt','Promt'],'prompt')
test['SmartScreen']= test['SmartScreen'].replace([np.nan,'&#x01;','&#x02;','Off','off','absent','OFF','0','of','Deny','ABSENT','ting','ExistsNotSet','Block','BLOCK','Not-there','&#x03;','00000000'],'NOT-THERE')
test['SmartScreen']= test['SmartScreen'].replace(['RequireAdmin','requireadmin','RequiredAdmin','Admin'],'ADMIN')
test['SmartScreen']= test['SmartScreen'].replace(['On','ON','on'],'oN')
test['SmartScreen']= test['SmartScreen'].replace(['Prompt','warn','Warn','Promprt'],'prompt')

#Let's take a look at the values

mini_train['SmartScreen'].value_counts()
#Dealing with Census_ProcessorClass missing values
mini_train['Census_ProcessorClass']=mini_train['Census_ProcessorClass'].replace(np.nan,'absent')
test['Census_ProcessorClass']=test['Census_ProcessorClass'].replace(np.nan,'absent')

#Let's take a look at the values
mini_train['Census_ProcessorClass'].value_counts()
#Dealing with Census Primary Disk Type Name
mini_train['Census_PrimaryDiskTypeName']=mini_train['Census_PrimaryDiskTypeName'].replace(['UNKNOWN','Unspecified'],'Generic')
test['Census_PrimaryDiskTypeName']=test['Census_PrimaryDiskTypeName'].replace(['UNKNOWN','Unspecified'],'Generic')

#Let's take a look at the values
mini_train['Census_PrimaryDiskTypeName'].value_counts()
#Dealing with Census_ChasisTypeName
mini_train['Census_ChassisTypeName']=mini_train['Census_ChassisTypeName'].replace([ 'MiniTower', 'Other', 'UNKNOWN',
                  'LowProfileDesktop', 'Detachable', 'HandHeld', 'SpaceSaving',
                  'Tablet', 'Tower', 'Unknown', 'MainServerChassis',
                  'LunchBox', 'MiniPC', 'RackMountChassis',
                  'BusExpansionChassis', 'SubNotebook', '30', '0', 'Blade',
                  '35', 'StickPC', 'SealedCasePC', 'SubChassis', 'PizzaBox',
                  '39', '36', '127', '81', 'BladeEnclosure', '112',
                  'CompactPCI', '49', '45', 'DockingStation', '76', '44',
                  'EmbeddedPC', '28', '25', '88', 'ExpansionChassis', '31',
                  '32', 'MultisystemChassis', 'IoTGateway', '82',np.nan],'OTHER')
test['Census_ChassisTypeName']=test['Census_ChassisTypeName'].replace([ 'MiniTower', 'Other', 'UNKNOWN',
                 'LowProfileDesktop', 'Detachable', 'HandHeld', 'SpaceSaving',
                  'Tablet', 'Tower', 'Unknown', 'MainServerChassis',
                  'LunchBox', 'MiniPC', 'RackMountChassis',
                  'BusExpansionChassis', 'SubNotebook', '30', '0', 'Blade',
                  '35', 'StickPC', 'SealedCasePC', 'SubChassis', 'PizzaBox',
                  '39', '36', '127', '81', 'BladeEnclosure', '112',
                  'CompactPCI', '49', '45', 'DockingStation', '76', '44',
                  'EmbeddedPC', '28', '25', '88', 'ExpansionChassis', '31',
                  '32', 'MultisystemChassis', 'IoTGateway', '82','120','93','64','84','120','PeripheralChassis','83',np.nan],'OTHER')

#Let's take a look at the values
mini_train['Census_ChassisTypeName'].value_counts()
#Dealing with Census_InternalBatteryType

mini_train['Census_InternalBatteryType']=mini_train['Census_InternalBatteryType'].replace(['li-i', '#', 'lip', 'liio', 'li p', 'li', 'nimh',
                  'real', 'bq20', 'pbac', 'vbox', 'lhp0', 'unkn', '4cel',
                  'lipo', 'lgi0', 'ithi', 'li-l', 'ram', '4ion', 'í-i',
                  '÷ÿóö', 'lio', 'cl53', 'ÿÿÿÿ', 'a132', 'd', 'virt', 'ca48',
                  'batt', 'asmb', 'bad', 'a140', 'lit', 'lio', 'lipp', 'liÿÿ',
                  '0x0b', 'lÿÿÿ', '3ion', '6ion', '4lio', 'lp', 'li?', 'ion',
                  'pbso', 'a138', 'li-h', '3500', 'ots0', 'h00j', 'li',
                  'sams', 'ip', '8', '#TAB#', 'l&#TAB#', 'lio', '@i',
                  'l', 'lgl0', 'lai0', 'lilo', 'pa50', 'h4°s', '5nm1', 'li-p',
                  'lhpo', '0ts0', 'pad0', 'sail', 'p-sn', 'icp3', 'a130',
                  '2337', '˙˙˙', 'lgs0',np.nan],'not-lion')

test['Census_InternalBatteryType']=test['Census_InternalBatteryType'].replace(['li-i', '#', 'lip', 'liio', 'li p', 'li', 'nimh',
                  'real', 'bq20', 'pbac', 'vbox', 'lhp0', 'unkn', '4cel',
                  'lipo', 'lgi0', 'ithi', 'li-l', 'ram', '4ion', 'í-i',
                  '÷ÿóö', 'lio', 'cl53', 'ÿÿÿÿ', 'a132', 'd', 'virt', 'ca48',
                  'batt', 'asmb', 'bad', 'a140', 'lit', 'lio', 'lipp', 'liÿÿ',
                  '0x0b', 'lÿÿÿ', '3ion', '6ion', '4lio', 'lp', 'li?', 'ion',
                  'pbso', 'a138', 'li-h', '3500', 'ots0', 'h00j', 'li',
                  'sams', 'ip', '8', '#TAB#', 'l&#TAB#', 'lio', '@i',
                  'l', 'lgl0', 'lai0', 'lilo', 'pa50', 'h4°s', '5nm1', 'li-p',
                  'lhpo', '0ts0', 'pad0', 'sail', 'p-sn', 'icp3', 'a130',
                  '2337', '˙˙˙', 'lgs0','2','l?', 'pcs0', 'ioon', 'pgd0', '9ion', '5ion','45n1', 'fake', 'nion', '@nn', 'loÿÿ', 'ñ±¿', 'lgco','l\x0b?','\x0c@nn',np.nan],'not-lion')

#Let's take a look at the values
mini_train['Census_InternalBatteryType'].value_counts()

#Looking at the data
mini_train.head()
#Let's start with EngineVersion

new_var =list()

a = list(mini_train['EngineVersion'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass
  

#Make the change
mini_train['EngineVersion']=pd.Series(new_var)

#Making the same change to the testing data

new_var =list()

a = list(test['EngineVersion'])

for i in range(len(a)):
  new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  

#Make the change
test['EngineVersion']=pd.Series(new_var)

#Next,let's go to AppVersion

new_var = list()

a = list(mini_train['AppVersion'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass
  
#Make the change
mini_train['AppVersion'] = pd.Series(new_var)
#Making the change to the testing data
new_var = list()

a = list(test['AppVersion'])

for i in range(len(a)):
  new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  
#Make the change
test['AppVersion'] = pd.Series(new_var)
#Next, let's go to AvSigVersion

new_var = list()

a = list(mini_train.loc[:,'AvSigVersion'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass


#Make the change
mini_train['AvSigVersion'] = pd.Series(new_var)
#Making the change to the testing data

new_var = list()

a = list(test.loc[:,'AvSigVersion'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass


#Make the change
test['AvSigVersion'] = pd.Series(new_var)
#Next, let's go to OsVer
new_var = list()
a = list(mini_train['OsVer'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass

#Make the change
mini_train['OsVer'] = pd.Series(new_var)
#Making the change to the testing data

new_var = list()
a = list(test['OsVer'])

for i in range(len(a)):
  new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))

#Make the change
test['OsVer'] = pd.Series(new_var)
#Next, let's go to OsVer
new_var = list()
a = list(mini_train['Census_OSVersion'])

for i in range(len(a)):
  try:
    new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))
  except:
    pass
  

#Make the change
mini_train['Census_OSVersion'] = pd.Series(new_var)
#Making the change to the testing data

new_var = list()
a = list(test['Census_OSVersion'])

for i in range(len(a)):
  new_var.append(np.array(a[i].split('.')[0]).astype(np.int)*(1+np.sum(np.array(a[i].split('.')[1:3]).astype(np.float)*np.array([1,1/100000]))))

#Make the change
test['Census_OSVersion'] = pd.Series(new_var)
#Next, let's go to OsVer
new_var = list()
a = list(mini_train['OsBuildLab'])

for i in range(len(a)):
  try:
    new_var.append(np.sum(np.array(a[i].split('.')[-1].split('-')).astype(np.int)*np.array([1,1/10000])))
  except:
    pass

#Make the change
mini_train['OsBuildLab'] = pd.Series(new_var)
#Making the change to the testing data

new_var = list()
a = list(test['OsBuildLab'])

for i in range(len(a)):
  try:
    new_var.append(np.sum(np.array(a[i].split('.')[-1].split('-')).astype(np.int)*np.array([1,1/10000])))
  except:
    pass

#Make the change
test['OsBuildLab'] = pd.Series(new_var)
#Let's these datasets to pickles
#mini_train.to_pickle("/content/drive/My Drive/Microsoft/train_reduced_1.pkl")
#test.to_pickle("/content/drive/My Drive/Microsoft/test_reduced_1.pkl")