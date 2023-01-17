import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split, ParameterGrid

from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss

import operator

import json

from IPython import display

import os

import warnings



np.random.seed(0)

warnings.filterwarnings("ignore")

THRESHOLD = 4
# read data from file

train = pd.read_csv("../input/padhai-module1-assignment/train.csv") 

test = pd.read_csv("../input/padhai-module1-assignment/test.csv")



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
def data_clean(data):

    

    # Let's first remove all missing value features

    columns_to_remove = ['Also Known As','Applications','Audio Features','Bezel-less display'

                         'Browser','Build Material','Co-Processor','Browser'

                         'Display Colour','Mobile High-Definition Link(MHL)',

                         'Music', 'Email','Fingerprint Sensor Position',

                         'Games','HDMI','Heart Rate Monitor','IRIS Scanner', 

                         'Optical Image Stabilisation','Other Facilities',

                         'Phone Book','Physical Aperture','Quick Charging',

                         'Ring Tone','Ruggedness','SAR Value','SIM 3','SMS',

                         'Screen Protection','Screen to Body Ratio (claimed by the brand)',

                         'Sensor','Software Based Aperture', 'Special Features',

                         'Standby time','Stylus','TalkTime', 'USB Type-C',

                         'Video Player', 'Video Recording Features','Waterproof',

                         'Wireless Charging','USB OTG Support', 'Video Recording','Java']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    #Features having very low variance 

    columns_to_remove = ['Architecture','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    # Multivalued:

    columns_to_remove = ['Architecture','Launch Date','Audio Jack','GPS','Loudspeaker','Network','Network Support','VoLTE', 'Custom UI']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    # Not much important

    columns_to_remove = ['Bluetooth', 'Settings','Wi-Fi','Wi-Fi Features']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]

    

    return data
train = data_clean(train)

test = data_clean(test)
train = train[(train.isnull().sum(axis=1) <= 15)]

# You shouldn't remove data points from test set

#test = test[(test.isnull().sum(axis=1) <= 15)]
# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
def for_integer(test):

    try:

        test = test.strip()

        return int(test.split(' ')[0])

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass



def for_string(test):

    try:

        test = test.strip()

        return (test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



def for_float(test):

    try:

        test = test.strip()

        return float(test.split(' ')[0])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass



    

def for_Internal_Memory(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[1] == 'GB':

            return int(test[0])

        if test[1] == 'MB':

#             print("here")

            return (int(test[0]) * 0.001)

    except IOError:

           pass

    except ValueError:

        pass

    except:

        pass

    

def find_freq(test):

    try:

        test = test.strip()

        test = test.split(' ')

        if test[2][0] == '(':

            return float(test[2][1:])

        return float(test[2])

    except IOError:

        pass

    except ValueError:

        pass

    except:

        pass

def data_clean_2(x):

    data = x.copy()

    

    data['Capacity'] = data['Capacity'].apply(for_integer)



    data['Height'] = data['Height'].apply(for_float)

    data['Height'] = data['Height'].fillna(data['Height'].mean())



    data['Internal Memory'] = data['Internal Memory'].apply(for_Internal_Memory)



    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)



    data['Internal Memory'] = data['Internal Memory'].fillna(data['Internal Memory'].median())

    data['Internal Memory'] = data['Internal Memory'].astype(int)



    data['RAM'] = data['RAM'].apply(for_integer)

    data['RAM'] = data['RAM'].fillna(data['RAM'].median())

    data['RAM'] = data['RAM'].astype(int)



    data['Resolution'] = data['Resolution'].apply(for_integer)

    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())

    data['Resolution'] = data['Resolution'].astype(int)



    data['Screen Size'] = data['Screen Size'].apply(for_float)



    data['Thickness'] = data['Thickness'].apply(for_float)

    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())

    data['Thickness'] = data['Thickness'].round(2)



    data['Type'] = data['Type'].fillna('Li-Polymer')



    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(for_float)

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(data['Screen to Body Ratio (calculated)'].mean())

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(2)



    data['Width'] = data['Width'].apply(for_float)

    data['Width'] = data['Width'].fillna(data['Width'].mean())

    data['Width'] = data['Width'].round(2)



    data['Flash'][data['Flash'].isna() == True] = "Other"



    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"



    data['Num_cores'] = data['Processor'].apply(for_string)

    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"





    data['Processor_frequency'] = data['Processor'].apply(find_freq)

    #because there is one entry with 208MHz values, to convert it to GHz

    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208

    data['Processor_frequency'] = data['Processor_frequency'].fillna(data['Processor_frequency'].mean())

    data['Processor_frequency'] = data['Processor_frequency'].round(2)



    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"



    #simplifyig Operating System to os_name for simplicity

    data['os_name'] = data['Operating System'].apply(for_string)

    data['os_name'][data['os_name'].isna() == True] = "Other"



    data['Sim1'] = data['SIM 1'].apply(for_string)



    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"



    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"



    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() == True] = "Other"



    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"



    data['Weight'] = data['Weight'].apply(for_integer)

    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())

    data['Weight'] = data['Weight'].astype(int)



    data['SIM 2'] = data['SIM 2'].apply(for_string)

    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"

    

    return data
train = data_clean_2(train)

test = data_clean_2(test)



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
def data_clean_3(x):

    

    data = x.copy()



    columns_to_remove = ['User Available Storage','SIM Size','Chipset','Processor','Autofocus','Aspect Ratio','Touch Screen',

                        'Bezel-less display','Operating System','SIM 1','USB Connectivity','Other Sensors','Graphics','FM Radio',

                        'NFC','Shooting Modes','Browser','Display Colour' ]



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = [ 'Screen Resolution','User Replaceable','Camera Features',

                        'Thickness', 'Display Type']



    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]





    columns_to_remove = ['Fingerprint Sensor', 'Flash', 'Rating Count', 'Review Count','Image Resolution','Type','Expandable Memory',\

                        'Colours','Width','Model']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))

    data = data[columns_to_retain]



    return data
train = data_clean_3(train)

test = data_clean_3(test)



# check the number of features and data points in train

print("Number of data points in train: %d" % train.shape[0])

print("Number of features in train: %d" % train.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test.shape[0])

print("Number of features in test: %d" % test.shape[1])
# one hot encoding



train_ids = train['PhoneId']

test_ids = test['PhoneId']



cols = list(test.columns)

cols.remove('PhoneId')

cols.insert(0, 'PhoneId')



combined = pd.concat([train.drop('Rating', axis=1)[cols], test[cols]])

print(combined.shape)

print(combined.columns)



combined = pd.get_dummies(combined)

print(combined.shape)

print(combined.columns)



train_new = combined[combined['PhoneId'].isin(train_ids)]

test_new = combined[combined['PhoneId'].isin(test_ids)]
train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')
# check the number of features and data points in train

print("Number of data points in train: %d" % train_new.shape[0])

print("Number of features in train: %d" % train_new.shape[1])



# check the number of features and data points in test

print("Number of data points in test: %d" % test_new.shape[0])

print("Number of features in test: %d" % test_new.shape[1])
train_new.to_csv('mobile_cleaned.csv', index = False)
train_new.head()
test_new.head()
train_new.describe
type(train_new)
def errorMB(x):

    if(x==512):

        return 0.5

    elif(x==256):

        return 0.25

    else:

        return x

train_new['RAM']=train_new['RAM'].apply(errorMB)

test_new['RAM']=test_new['RAM'].apply(errorMB)



train_new['Internal Memory']=train_new['Internal Memory'].apply(errorMB)

test_new['Internal Memory']=test_new['Internal Memory'].apply(errorMB)

max(train_new['Height'])
X_train=train_new.drop(['PhoneId','Rating','Brand_10.or', 'Brand_Apple',

       'Brand_Asus', 'Brand_Billion', 'Brand_Blackberry', 'Brand_Comio',

       'Brand_Coolpad', 'Brand_Do', 'Brand_Gionee', 'Brand_Google',

       'Brand_HTC', 'Brand_Honor', 'Brand_Huawei', 'Brand_InFocus',

       'Brand_Infinix', 'Brand_Intex', 'Brand_Itel', 'Brand_Jivi',

       'Brand_Karbonn', 'Brand_LG', 'Brand_Lava', 'Brand_LeEco',

       'Brand_Lenovo', 'Brand_Lephone', 'Brand_Lyf', 'Brand_Meizu',

       'Brand_Micromax', 'Brand_Mobiistar', 'Brand_Moto', 'Brand_Motorola',

       'Brand_Nokia', 'Brand_Nubia', 'Brand_OPPO', 'Brand_OnePlus',

       'Brand_Oppo', 'Brand_Panasonic', 'Brand_Razer', 'Brand_Realme',

       'Brand_Reliance', 'Brand_Samsung', 'Brand_Sony', 'Brand_Spice',

       'Brand_Tecno', 'Brand_Ulefone', 'Brand_VOTO', 'Brand_Vivo',

       'Brand_Xiaomi', 'Brand_Xiaomi Poco', 'Brand_Yu', 'Brand_iVooMi',

       'Sim1_2G', 'Sim1_3G', 'Sim1_4G', 'Num_cores_312', 'Num_cores_Deca',

       'Num_cores_Dual', 'Num_cores_Hexa', 'Num_cores_Octa', 'Num_cores_Other',

       'Num_cores_Quad', 'Num_cores_Tru-Octa',

       'SIM Slot(s)_Dual SIM, GSM+CDMA', 'SIM Slot(s)_Dual SIM, GSM+GSM',

       'SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE',

       'SIM Slot(s)_Single SIM, GSM', 'SIM 2_2G', 'SIM 2_3G', 'SIM 2_4G',

       'SIM 2_Other', 'os_name_Android', 'os_name_Blackberry', 'os_name_KAI',

       'os_name_Nokia', 'os_name_Other', 'os_name_Tizen', 'os_name_iOS'],axis=1)

X_test=test_new.drop(['PhoneId','Brand_10.or', 'Brand_Apple',

       'Brand_Asus', 'Brand_Billion', 'Brand_Blackberry', 'Brand_Comio',

       'Brand_Coolpad', 'Brand_Do', 'Brand_Gionee', 'Brand_Google',

       'Brand_HTC', 'Brand_Honor', 'Brand_Huawei', 'Brand_InFocus',

       'Brand_Infinix', 'Brand_Intex', 'Brand_Itel', 'Brand_Jivi',

       'Brand_Karbonn', 'Brand_LG', 'Brand_Lava', 'Brand_LeEco',

       'Brand_Lenovo', 'Brand_Lephone', 'Brand_Lyf', 'Brand_Meizu',

       'Brand_Micromax', 'Brand_Mobiistar', 'Brand_Moto', 'Brand_Motorola',

       'Brand_Nokia', 'Brand_Nubia', 'Brand_OPPO', 'Brand_OnePlus',

       'Brand_Oppo', 'Brand_Panasonic', 'Brand_Razer', 'Brand_Realme',

       'Brand_Reliance', 'Brand_Samsung', 'Brand_Sony', 'Brand_Spice',

       'Brand_Tecno', 'Brand_Ulefone', 'Brand_VOTO', 'Brand_Vivo',

       'Brand_Xiaomi', 'Brand_Xiaomi Poco', 'Brand_Yu', 'Brand_iVooMi',

       'Sim1_2G', 'Sim1_3G', 'Sim1_4G', 'Num_cores_312', 'Num_cores_Deca',

       'Num_cores_Dual', 'Num_cores_Hexa', 'Num_cores_Octa', 'Num_cores_Other',

       'Num_cores_Quad', 'Num_cores_Tru-Octa',

       'SIM Slot(s)_Dual SIM, GSM+CDMA', 'SIM Slot(s)_Dual SIM, GSM+GSM',

       'SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE',

       'SIM Slot(s)_Single SIM, GSM', 'SIM 2_2G', 'SIM 2_3G', 'SIM 2_4G',

       'SIM 2_Other', 'os_name_Android', 'os_name_Blackberry', 'os_name_KAI',

       'os_name_Nokia', 'os_name_Other', 'os_name_Tizen', 'os_name_iOS'],axis=1)

Y_train=train_new['Rating']
scaler=MinMaxScaler()

X_normalized_train=scaler.fit_transform(X_train)

plt.plot(X_normalized_train.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_normalized_test=scaler.fit_transform(X_test)
plt.plot(X_normalized_test.T,'*')

plt.xticks(rotation='vertical')

plt.show()
X_normalized_train=pd.DataFrame.from_records(X_normalized_train)

X_normalized_test=pd.DataFrame.from_records(X_normalized_test)



#X_binarised_train=X_normalized_train.apply(pd.cut,bins=2,labels=[0,1])

#X_binarised_test=X_normalized_test.apply(pd.cut,bins=2,labels=[0,1])



#X_binarised_train=X_binarised_train.values

#X_binarised_test=X_binarised_test.values
print(X_normalized_train)
X_normalized_train.mean()
def res(x):

    return 1 if(x>=0.3) else 0

def cap(x):

    return 1 if(x>=0.15) else 0

def pxd(x):

    return 1 if(x>=0.4) else 0

def wgt(x):

    return 1 if(x==0) else 0

def stb(x):

    return 1 if(x>=0.8) else 0

def ram(x):

    return 1 if(x>=0.023) else 0

def mem(x):

    return 1 if(x>=0.123) else 0

def pro(x):

    return 1 if(x>=0.4) else 0

def scr(x):

    return 1 if(x>=0.73) else 0

def hgt(x):

    return 1 if(x>=0.58) else 0





X_binarised_train=X_normalized_train

X_binarised_test=X_normalized_test



X_binarised_train[0]=X_binarised_train[0].apply(res)

X_binarised_test[0]=X_binarised_test[0].apply(res)



X_binarised_train[1]=X_binarised_train[1].apply(cap)

X_binarised_test[1]=X_binarised_test[1].apply(cap)



X_binarised_train[2]=X_binarised_train[2].apply(pxd)

X_binarised_test[2]=X_binarised_test[2].apply(pxd)



X_binarised_train[3]=X_binarised_train[3].apply(wgt)

X_binarised_test[3]=X_binarised_test[3].apply(wgt)



X_binarised_train[4]=X_binarised_train[4].apply(stb)

X_binarised_test[4]=X_binarised_test[4].apply(stb)



X_binarised_train[5]=X_binarised_train[5].apply(ram)

X_binarised_test[5]=X_binarised_test[5].apply(ram)



X_binarised_train[6]=X_binarised_train[6].apply(mem)

X_binarised_test[6]=X_binarised_test[6].apply(mem)



X_binarised_train[7]=X_binarised_train[7].apply(pro)

X_binarised_test[7]=X_binarised_test[7].apply(pro)



X_binarised_train[8]=X_binarised_train[8].apply(scr)

X_binarised_test[8]=X_binarised_test[8].apply(scr)



X_binarised_train[9]=X_binarised_train[9].apply(hgt)

X_binarised_test[9]=X_binarised_test[9].apply(hgt)
X_binarised_train=X_binarised_train.values

X_binarised_test=X_binarised_test.values
Y_binarised_train=[]

for y in Y_train:

    if(y>=THRESHOLD):

        Y_binarised_train.append(1)

    else:

        Y_binarised_train.append(0)
Y_binarised_train=pd.Series(Y_binarised_train)

print(Y_binarised_train)
train_diag=pd.DataFrame.from_records(X_binarised_train)
train_diag['Rating']=Y_binarised_train
print(train_diag)
train_diag.groupby('Rating').mean()
train_diag.mean()
class MPNeuron:

    

    def __init__(self):

        self.b=None

        

    def model(self,x):

        return (sum(x)>=self.b)

              

    def predict(self,X):

        Y=[]

        for x in X:

            result=self.model(x)

            Y.append(result)

        return np.array(Y)

    

    def fit(self,X,Y):

        

        accuracy={}

        

        for b in range(X.shape[1]+1):

            self.b=b

            Y_pred=self.predict(X)

            accuracy[b]=accuracy_score(Y_pred,Y)

            

        lists=sorted(accuracy.items())

        x,y=zip(*lists)

        plt.plot(x,y,'*')

        plt.show()

        

        best_b=max(accuracy,key=accuracy.get)

        self.b=best_b

        

        print("Optimal threshold for given data is ",best_b)

        print("Highest accuracy achieved is ",accuracy[best_b])

        
mpneuron=MPNeuron()
mpneuron.fit(X_binarised_train,Y_binarised_train)
Y_predicted_test=mpneuron.predict(X_binarised_test)
Y_predicted_test=pd.DataFrame(Y_predicted_test,columns=['Rating'],dtype='int')

print(Y_predicted_test)
submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':Y_predicted_test['Rating']})

submission = submission[['PhoneId', 'Class']]

submission.head()
submission.to_csv("submission.csv", index=False)