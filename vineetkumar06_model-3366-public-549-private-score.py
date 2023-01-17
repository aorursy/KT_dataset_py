import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

from sklearn.utils import resample

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/ml-analaytic2/train_u5jK80M/train.csv')

test=pd.read_csv('../input/ml-analaytic2/test_3BA6GZX/test.csv')

sub=pd.read_csv('../input/ml-analaytic2/sample_submission_1Sfyqeb/sample_submission.csv')



# feture engineering

train['origination_date'] = pd.to_datetime(train['origination_date'])

train['first_payment_date'] = pd.to_datetime(train['first_payment_date'])

def ever_previous_deliquenc(x):

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]!=0:

            return 1

    return 0

train['ever_previous_deliquenc']=train.apply(ever_previous_deliquenc, axis=1)

train['loan_debt_ratio']=train['loan_to_value']/train['debt_to_income_ratio']

test['ever_previous_deliquenc']=test.apply(ever_previous_deliquenc, axis=1)

test['loan_debt_ratio']=test['loan_to_value']/test['debt_to_income_ratio']



def ever_2consecutive_deliquenc(x):

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]>1:

            return 1

    return 0

def ever_3consecutive_deliquenc(x):

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]>2:

            return 1

    return 0

train['ever_2consecutive_deliquenc']=train.apply(ever_2consecutive_deliquenc, axis=1)

test['ever_2consecutive_deliquenc']=test.apply(ever_2consecutive_deliquenc, axis=1)

train['ever_3consecutive_deliquenc']=train.apply(ever_3consecutive_deliquenc, axis=1)

test['ever_3consecutive_deliquenc']=test.apply(ever_3consecutive_deliquenc, axis=1)

def is_3frequent_deliquenc(x):

    count=0

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]>=1:

            count=count+1

            if(count>=3):

                return 1

    return 0



def is_4frequent_deliquenc(x):

    count=0

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]>=1:

            count=count+1

            if(count>=4):

                return 1

    return 0

def is_5frequent_deliquenc(x):

    count=0

    for i in ['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]>=1:

            count=count+1

            if(count>=5):

                return 1

    return 0

def is_past3month_deliquenc_clear(x):

    count=0

    for i in ['m10','m11','m12']:

        if x[i]==0:

            count=count+1

    if(count==3):

        return 1

    return 0

def is_past6month_deliquenc_clear(x):

    count=0

    for i in ['m7','m8','m9','m10','m11','m12']:

        if x[i]==0:

            count=count+1

    if(count==6):

        return 1

    return 0

def is_past9month_deliquenc_clear(x):

    count=0

    for i in ['m4','m5','m6','m7','m8','m9','m10','m11','m12']:

        if x[i]==0:

            count=count+1

    if(count==9):

        return 1

    return 0

train['is_3frequent_deliquenc']=train.apply(is_3frequent_deliquenc, axis=1)

test['is_3frequent_deliquenc']=test.apply(is_3frequent_deliquenc, axis=1)

train['is_4frequent_deliquenc']=train.apply(is_4frequent_deliquenc, axis=1)

test['is_4frequent_deliquenc']=test.apply(is_4frequent_deliquenc, axis=1)

train['is_5frequent_deliquenc']=train.apply(is_5frequent_deliquenc, axis=1)

test['is_5frequent_deliquenc']=test.apply(is_5frequent_deliquenc, axis=1)

train['is_past3month_deliquenc_clear']=train.apply(is_past3month_deliquenc_clear, axis=1)

test['is_past3month_deliquenc_clear']=test.apply(is_past3month_deliquenc_clear, axis=1)

train['is_past6month_deliquenc_clear']=train.apply(is_past6month_deliquenc_clear, axis=1)

test['is_past6month_deliquenc_clear']=test.apply(is_past6month_deliquenc_clear, axis=1)

train['is_past9month_deliquenc_clear']=train.apply(is_past9month_deliquenc_clear, axis=1)

test['is_past9month_deliquenc_clear']=test.apply(is_past9month_deliquenc_clear, axis=1)

# creating average deliquence frequency for bank

tt=pd.DataFrame(train.groupby(['financial_institution'])['m13'].value_counts())

tt.rename(columns={'m13':'count'},inplace=True)

tt=tt.reset_index()

institute_ratio = pd.DataFrame(columns = ['financial_institution', 'deliquenc_frequency'])

institure=train['financial_institution'].unique()

for i in institure:

    fname=i

    ft=tt[tt['financial_institution']==i]

    if len(ft)==2:

        ratio=ft[ft['m13']==1]['count'].values/ft[ft['m13']==0]['count'].values

        ratio=ratio[0]

    else:

        if len(ft[ft['m13']==0]['count'])==1:

            ratio= 0

    institute_ratio=institute_ratio.append({'financial_institution':i, 'deliquenc_frequency':ratio}, ignore_index=True)



train = pd.merge(train, institute_ratio,on='financial_institution')

test=pd.merge(test, institute_ratio,on='financial_institution')

#encoding the categorical (Alphabetical) feature

le = preprocessing.LabelEncoder()

data=pd.concat([train, test])

for i in ['source','financial_institution','loan_purpose']:

    source_l=le.fit(data[i])

    train[i]=source_l.transform(train[i])

    test[i]=source_l.transform(test[i])

train['monthly_interest']=(train['interest_rate']/12)*(train['unpaid_principal_bal']/100)

test['monthly_interest']=(test['interest_rate']/12)*(test['unpaid_principal_bal']/100)

data['monthly_interest']=(data['interest_rate']/12)*(data['unpaid_principal_bal']/100)

train.head()

categorical_collumn=['source','financial_institution','is_past6month_deliquenc_clear','is_past9month_deliquenc_clear','is_past3month_deliquenc_clear','is_3frequent_deliquenc','is_4frequent_deliquenc','is_5frequent_deliquenc','number_of_borrowers','loan_purpose','insurance_type','ever_3consecutive_deliquenc','ever_2consecutive_deliquenc','ever_previous_deliquenc','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']

column_to_scale=['interest_rate','unpaid_principal_bal','deliquenc_frequency','loan_term','loan_to_value','debt_to_income_ratio','borrower_credit_score','monthly_interest','loan_debt_ratio','insurance_percent','co-borrower_credit_score']

#categorical_collumn=['source','financial_institution','number_of_borrowers','loan_purpose','insurance_type','m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','m11','m12']

#column_to_scale=['interest_rate','unpaid_principal_bal','loan_term','loan_to_value','debt_to_income_ratio','borrower_credit_score','insurance_percent','co-borrower_credit_score']



del train['origination_date']

del test['origination_date']

del train['first_payment_date']

del test['first_payment_date']

subdf = pd.DataFrame()

subdf['loan_id']=test['loan_id']

del test['loan_id'] 
#scaling the features to same range

std_scaler = StandardScaler()

for i in column_to_scale:

    ss = StandardScaler().fit(data[i].values.reshape(-1, 1))

    train[i]=ss.transform(train[i].values.reshape(-1, 1))

    test[i]=ss.transform(test[i].values.reshape(-1, 1))
print('No Delinquency', round(train['m13'].value_counts()[0]/len(train) * 100,2), '% of the dataset')

print('Delinquency', round(train['m13'].value_counts()[1]/len(train) * 100,2), '% of the dataset')
X = train.drop('m13', axis=1)

y = train['m13']

sss = StratifiedKFold(n_splits=5, random_state=1533, shuffle=False)

for train_index, test_index in sss.split(X, y):

    print("Train:", train_index, "Test:", test_index)

    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]

    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]

    

del original_Xtrain['loan_id']

del original_Xtest['loan_id']

clist=[original_Xtrain.columns.get_loc(c) for c in categorical_collumn if c in original_Xtrain]
# concatenate our training data back together

X = pd.concat([original_Xtrain, original_ytrain], axis=1)

X.head()


# separate minority and majority classes

not_Delinquency = X[X.m13==0]

Delinquency = X[X.m13==1]



# upsample minority

Delinquency_upsampled = resample(Delinquency,

                                  replace=True, # sample with replacement

                                      n_samples=1908, # match number in majority class

                                  random_state=1533) # reproducible results

 # downsample majority

Delinquency_downsampled = resample(not_Delinquency,

                                        replace = False, # sample without replacement

                                        n_samples = 60000, # match minority n

                                        random_state = 1533) # reproducible results

upsampled = pd.concat([Delinquency_downsampled, Delinquency_upsampled])

        # check new class counts

upsampled.m13.value_counts()

original_Xtrain=upsampled.drop('m13', axis=1)

original_ytrain=upsampled['m13']
# train model

rfc = RandomForestClassifier(n_estimators=69,max_depth=7,random_state=1533).fit(original_Xtrain, original_ytrain)



# predict on test set

rfc_pred = rfc.predict(original_Xtest)



print(f1_score(original_ytest, rfc_pred))

print(classification_report(original_ytest, rfc_pred))
model = AdaBoostClassifier(RandomForestClassifier(n_estimators=69,max_depth=7,random_state=1533),

                         algorithm="SAMME",n_estimators=3,random_state=1533).fit(original_Xtrain, original_ytrain)

rfc_pred = model.predict(original_Xtest)

f1_score(original_ytest, rfc_pred)
print( sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), original_Xtrain.columns), reverse=True))
test_pred = model.predict(test)

subdf['m13']=test_pred

subdf['m13'].value_counts()

subdf.to_csv('model_sol.csv',index=False)