# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/application_train.csv")
test = pd.read_csv("../input/application_test.csv")
bureau = pd.read_csv("../input/bureau.csv")
bureau_bal = pd.read_csv("../input/bureau_balance.csv")
credit_card_bal = pd.read_csv("../input/credit_card_balance.csv")
installment_payments = pd.read_csv("../input/installments_payments.csv")
pos_cash_bal = pd.read_csv("../input/POS_CASH_balance.csv")
previous_application = pd.read_csv("../input/previous_application.csv")
sample_application = pd.read_csv("../input/sample_submission.csv")
trainCopy = train.copy()
testCopy = test.copy()
bureauCopy = bureau.copy()
bureau_balCopy = bureau_bal.copy()
credit_card_balCopy = credit_card_bal.copy()
installment_paymentsCopy = installment_payments.copy()
pos_cash_balCopy = pos_cash_bal.copy()
previous_applicationCopy = previous_application.copy()
previous_applicationCopy.head()
bureauCopy = bureauCopy.drop('CREDIT_ACTIVE', axis = 1)
bureauCopy = bureauCopy.drop('CREDIT_CURRENCY', axis = 1)
bureauCopy = bureauCopy.drop('CREDIT_TYPE', axis = 1)
bureau_summarise = bureauCopy.groupby(['SK_ID_CURR']).agg([np.min, np.max, np.mean]).reset_index()
bureau_summarise.columns = ['_'.join(col).strip() for col in bureau_summarise.columns.values]
bureau_summarise = bureau_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
bureau_summarise.head()
credit_card_balCopy =  credit_card_balCopy.drop('NAME_CONTRACT_STATUS', axis = 1)
credit_card_balCopy = credit_card_bal.drop('SK_ID_PREV', axis = 1)
credit_card_bal_summarise = credit_card_bal.groupby(['SK_ID_CURR']).agg([np.mean, np.min, np.max]).reset_index()
credit_card_bal_summarise.columns = ['_'.join(col).strip() for col in credit_card_bal_summarise.columns.values]
credit_card_bal_summarise = credit_card_bal_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
credit_card_bal_summarise.head()
installment_paymentsCopy = installment_paymentsCopy.drop('SK_ID_PREV', axis=1)
installment_payments_summarise = installment_paymentsCopy.groupby(['SK_ID_CURR']).agg([np.mean, np.min, np.max]).reset_index()
installment_payments_summarise.columns = ['_'.join(col).strip() for col in installment_payments_summarise.columns.values]
installment_payments_summarise = installment_payments_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
installment_payments_summarise.head()
pos_cash_balCopy =  pos_cash_balCopy.drop('NAME_CONTRACT_STATUS', axis = 1)
pos_cash_balCopy = pos_cash_balCopy.drop('SK_ID_PREV', axis=1)
pos_cash_bal_summarise = pos_cash_balCopy.groupby(['SK_ID_CURR']).agg([np.mean, np.min, np.max]).reset_index()
pos_cash_bal_summarise.columns = ['_'.join(col).strip() for col in pos_cash_bal_summarise.columns.values]
pos_cash_bal_summarise = pos_cash_bal_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
pos_cash_bal_summarise.head()
curr_bureau_map_id = bureau.loc[:,['SK_ID_CURR', 'SK_ID_BUREAU']]
curr_bureau_map_id = curr_bureau_map_id.drop_duplicates()
curr_bureau_map_id.head()
bureau_bal_summarise = bureau_balCopy.groupby(['SK_ID_BUREAU']).aggregate([np.min,np.max,np.mean]).reset_index()
bureau_bal_summarise.columns = ['_'.join(col).strip() for col in bureau_bal_summarise.columns.values]
bureau_bal_summarise = bureau_bal_summarise.rename(columns={'SK_ID_BUREAU_':'SK_ID_BUREAU'})
bureau_bal_summarise = bureau_bal_summarise.merge(curr_bureau_map_id, on = "SK_ID_BUREAU", how = "inner")
bureau_bal_summarise = bureau_bal_summarise.drop('SK_ID_BUREAU', axis = 1)
bureau_bal_summarise = bureau_bal_summarise.groupby(['SK_ID_CURR']).aggregate([np.mean]).reset_index()
bureau_bal_summarise.columns = ['_'.join(col).strip() for col in bureau_bal_summarise.columns.values]
bureau_bal_summarise = bureau_bal_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
bureau_bal_summarise.head()
previous_applicationCopy = previous_applicationCopy.drop('SK_ID_PREV', axis = 1)
previous_applicationCopy = previous_applicationCopy.drop('NAME_CONTRACT_TYPE', axis = 1)
previous_applicationCopy = previous_applicationCopy.drop('WEEKDAY_APPR_PROCESS_START', axis = 1)
previous_applicationCopy = previous_applicationCopy.drop('NAME_SELLER_INDUSTRY', axis = 1)
previous_applicationCopy = previous_applicationCopy.drop('NAME_YIELD_GROUP', axis = 1)
previous_applicationCopy = previous_applicationCopy.drop('PRODUCT_COMBINATION', axis = 1)
previous_application_summarise = previous_applicationCopy.groupby(['SK_ID_CURR']).aggregate([np.min,np.max,np.mean]).reset_index()
previous_application_summarise.columns = ['_'.join(col).strip() for col in previous_application_summarise.columns.values]
previous_application_summarise = previous_application_summarise.rename(columns={'SK_ID_CURR_':'SK_ID_CURR'})
previous_application_summarise.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

trainCopy['NAME_CONTRACT_TYPE'] = trainCopy['NAME_CONTRACT_TYPE'].fillna("Unknown")
le.fit(trainCopy['NAME_CONTRACT_TYPE'])
le.classes_
trainCopy['NAME_CONTRACT_TYPE'] = le.transform(trainCopy['NAME_CONTRACT_TYPE'])

trainCopy['CODE_GENDER'] = trainCopy['CODE_GENDER'].fillna("Unknown")
le.fit(trainCopy['CODE_GENDER'])
le.classes_
trainCopy['CODE_GENDER'] = le.transform(trainCopy['CODE_GENDER'])

trainCopy['FLAG_OWN_CAR'] = trainCopy['FLAG_OWN_CAR'].fillna("Unknown")
le.fit(trainCopy['FLAG_OWN_CAR'])
le.classes_
trainCopy['FLAG_OWN_CAR'] = le.transform(trainCopy['FLAG_OWN_CAR'])

trainCopy['FLAG_OWN_REALTY'] = trainCopy['FLAG_OWN_REALTY'].fillna("Unknown")
le.fit(trainCopy['FLAG_OWN_REALTY'])
le.classes_
trainCopy['FLAG_OWN_REALTY'] = le.transform(trainCopy['FLAG_OWN_REALTY'])


trainCopy['NAME_TYPE_SUITE'] = trainCopy['NAME_TYPE_SUITE'].fillna("Unknown")
le.fit(trainCopy['NAME_TYPE_SUITE'])
le.classes_
trainCopy['NAME_TYPE_SUITE'] = le.transform(trainCopy['NAME_TYPE_SUITE'])

trainCopy['NAME_INCOME_TYPE'] = trainCopy['NAME_INCOME_TYPE'].fillna("Unknown")
le.fit(trainCopy['NAME_INCOME_TYPE'])
le.classes_
trainCopy['NAME_INCOME_TYPE'] = le.transform(trainCopy['NAME_INCOME_TYPE'])

trainCopy['NAME_EDUCATION_TYPE'] = trainCopy['NAME_EDUCATION_TYPE'].fillna("Unknown")
le.fit(trainCopy['NAME_EDUCATION_TYPE'])
le.classes_
trainCopy['NAME_EDUCATION_TYPE'] = le.transform(trainCopy['NAME_EDUCATION_TYPE'])

trainCopy['NAME_FAMILY_STATUS'] = trainCopy['NAME_FAMILY_STATUS'].fillna("Unknown")
le.fit(trainCopy['NAME_FAMILY_STATUS'])
le.classes_
trainCopy['NAME_FAMILY_STATUS'] = le.transform(trainCopy['NAME_FAMILY_STATUS'])

trainCopy['NAME_HOUSING_TYPE'] = trainCopy['NAME_HOUSING_TYPE'].fillna("Unknown")
le.fit(trainCopy['NAME_HOUSING_TYPE'])
le.classes_
trainCopy['NAME_HOUSING_TYPE'] = le.transform(trainCopy['NAME_HOUSING_TYPE'])

trainCopy['OCCUPATION_TYPE'] = trainCopy['OCCUPATION_TYPE'].fillna("Unknown")
le.fit(trainCopy['OCCUPATION_TYPE'])
le.classes_
trainCopy['OCCUPATION_TYPE'] = le.transform(trainCopy['OCCUPATION_TYPE'])

trainCopy['ORGANIZATION_TYPE'] = trainCopy['ORGANIZATION_TYPE'].fillna("Unknown")
le.fit(trainCopy['ORGANIZATION_TYPE'])
le.classes_
trainCopy['ORGANIZATION_TYPE'] = le.transform(trainCopy['ORGANIZATION_TYPE'])

trainCopy['WEEKDAY_APPR_PROCESS_START'] = trainCopy['WEEKDAY_APPR_PROCESS_START'].fillna("Unknown")
le.fit(trainCopy['WEEKDAY_APPR_PROCESS_START'])
le.classes_
trainCopy['WEEKDAY_APPR_PROCESS_START'] = le.transform(trainCopy['WEEKDAY_APPR_PROCESS_START'])

trainCopy['FONDKAPREMONT_MODE'] = trainCopy['FONDKAPREMONT_MODE'].fillna("Unknown")
le.fit(trainCopy['FONDKAPREMONT_MODE'])
le.classes_
trainCopy['FONDKAPREMONT_MODE'] = le.transform(trainCopy['FONDKAPREMONT_MODE'])

trainCopy['HOUSETYPE_MODE'] = trainCopy['HOUSETYPE_MODE'].fillna("Unknown")
le.fit(trainCopy['HOUSETYPE_MODE'])
le.classes_
trainCopy['HOUSETYPE_MODE'] = le.transform(trainCopy['HOUSETYPE_MODE'])

trainCopy['WALLSMATERIAL_MODE'] = trainCopy['WALLSMATERIAL_MODE'].fillna("Unknown")
le.fit(trainCopy['WALLSMATERIAL_MODE'])
le.classes_
trainCopy['WALLSMATERIAL_MODE'] = le.transform(trainCopy['WALLSMATERIAL_MODE'])

trainCopy['EMERGENCYSTATE_MODE'] = trainCopy['EMERGENCYSTATE_MODE'].fillna("Unknown")
le.fit(trainCopy['EMERGENCYSTATE_MODE'])
le.classes_
trainCopy['EMERGENCYSTATE_MODE'] = le.transform(trainCopy['EMERGENCYSTATE_MODE'])
bureau_encoding = bureau.loc[:,["SK_ID_CURR", "CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"]]
CREDIT_ACTIVE = pd.get_dummies(bureau_encoding["CREDIT_ACTIVE"], prefix="CREDIT_ACTIVE")
bureau_encoding = bureau_encoding.drop('CREDIT_ACTIVE', axis = 1)
bureau = bureau.drop('CREDIT_ACTIVE', axis = 1)
bureau_encoding = pd.concat([bureau_encoding, CREDIT_ACTIVE], axis = 1)
CREDIT_CURRENCY = pd.get_dummies(bureau_encoding["CREDIT_CURRENCY"], prefix="CREDIT_CURRENCY")
bureau_encoding = bureau_encoding.drop('CREDIT_CURRENCY', axis = 1)
bureau = bureau.drop('CREDIT_CURRENCY', axis = 1)
bureau_encoding = pd.concat([bureau_encoding, CREDIT_CURRENCY], axis = 1)
CREDIT_TYPE = pd.get_dummies(bureau_encoding["CREDIT_TYPE"], prefix="CREDIT_TYPE")
bureau_encoding = bureau_encoding.drop('CREDIT_TYPE', axis = 1)
bureau = bureau.drop('CREDIT_TYPE', axis = 1)
bureau_encoding = pd.concat([bureau_encoding, CREDIT_TYPE], axis = 1)
bureau_encoding_summarise = bureau_encoding.groupby(['SK_ID_CURR']).sum().reset_index()
bureau_encoding.head()
bureau_bal_encoding = bureau_bal.loc[:, ["SK_ID_BUREAU","STATUS"]]
STATUS = pd.get_dummies(bureau_bal_encoding["STATUS"], prefix="STATUS")
bureau_bal_encoding = bureau_bal_encoding.drop('STATUS', axis = 1)
bureau_bal_encoding = pd.concat([bureau_bal_encoding, STATUS], axis = 1)
#bureau_bal_encoding.head()
bureau_bal_encoding_summaries = bureau_bal_encoding.groupby(['SK_ID_BUREAU']).sum().reset_index()
#bureau_bal_encoding_summaries.head()
bureau_bal_encoding_summaries = bureau_bal_encoding_summaries.merge(curr_bureau_map_id, on = "SK_ID_BUREAU", how = "left")
bureau_bal_encoding_summaries = bureau_bal_encoding_summaries.drop('SK_ID_BUREAU', axis = 1)
bureau_bal_encoding_summaries = bureau_bal_encoding_summaries.groupby(['SK_ID_CURR']).mean().reset_index()
bureau_bal_encoding_summaries.head()
curr_bureau_map_id['SK_ID_BUREAU'] = curr_bureau_map_id['SK_ID_BUREAU'].astype(str)
curr_bureau_map_id['SK_ID_CURR'] = curr_bureau_map_id['SK_ID_CURR'].astype(str)
credit_card_bal_encoding = credit_card_bal.loc[:, ["SK_ID_CURR","NAME_CONTRACT_STATUS"]]
NAME_CONTRACT_STATUS = pd.get_dummies(credit_card_bal_encoding["NAME_CONTRACT_STATUS"], prefix="NAME_CONTRACT_STATUS_CC")
credit_card_bal_encoding = credit_card_bal_encoding.drop('NAME_CONTRACT_STATUS', axis = 1)
credit_card_bal_encoding = pd.concat([credit_card_bal_encoding, NAME_CONTRACT_STATUS], axis = 1)
credit_card_bal_encoding.head()
credit_card_bal_encoding_summaries = credit_card_bal_encoding.groupby(['SK_ID_CURR']).sum().reset_index()
pos_cash_bal_encoding = pos_cash_bal.loc[:, ["SK_ID_CURR","NAME_CONTRACT_STATUS"]]
NAME_CONTRACT_STATUS = pd.get_dummies(pos_cash_bal_encoding["NAME_CONTRACT_STATUS"], prefix="NAME_CONTRACT_STATUS_POS")
pos_cash_bal_encoding = pos_cash_bal_encoding.drop('NAME_CONTRACT_STATUS', axis = 1)
pos_cash_bal_encoding = pd.concat([pos_cash_bal_encoding, NAME_CONTRACT_STATUS], axis = 1)
pos_cash_bal_encoding_summarise =  pos_cash_bal_encoding.groupby(['SK_ID_CURR']).sum().reset_index()
pos_cash_bal_encoding_summarise.head()
previous_applicationEncoding = previous_application.loc[:,['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'FLAG_LAST_APPL_PER_CONTRACT', 'WEEKDAY_APPR_PROCESS_START',
                                                          'NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE', 'CODE_REJECT_REASON',
                                                          'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO','NAME_PRODUCT_TYPE',
                                                          'CHANNEL_TYPE', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION', 'NAME_SELLER_INDUSTRY']]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

previous_applicationEncoding['NAME_CONTRACT_TYPE'] = previous_applicationEncoding['NAME_CONTRACT_TYPE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_CONTRACT_TYPE'])
le.classes_
previous_applicationEncoding['PREV_NAME_CONTRACT_TYPE'] = le.transform(previous_applicationEncoding['NAME_CONTRACT_TYPE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_CONTRACT_TYPE', axis = 1)

previous_applicationEncoding['FLAG_LAST_APPL_PER_CONTRACT'] = previous_applicationEncoding['FLAG_LAST_APPL_PER_CONTRACT'].fillna("Unknown")
le.fit(previous_applicationEncoding['FLAG_LAST_APPL_PER_CONTRACT'])
le.classes_
previous_applicationEncoding['PREV_FLAG_LAST_APPL_PER_CONTRACT'] = le.transform(previous_applicationEncoding['FLAG_LAST_APPL_PER_CONTRACT'])
previous_applicationEncoding = previous_applicationEncoding.drop('FLAG_LAST_APPL_PER_CONTRACT', axis = 1)

previous_applicationEncoding['WEEKDAY_APPR_PROCESS_START'] = previous_applicationEncoding['WEEKDAY_APPR_PROCESS_START'].fillna("Unknown")
le.fit(previous_applicationEncoding['WEEKDAY_APPR_PROCESS_START'])
le.classes_
previous_applicationEncoding['PREV_WEEKDAY_APPR_PROCESS_START'] = le.transform(previous_applicationEncoding['WEEKDAY_APPR_PROCESS_START'])
previous_applicationEncoding = previous_applicationEncoding.drop('WEEKDAY_APPR_PROCESS_START', axis = 1)

previous_applicationEncoding['NAME_CASH_LOAN_PURPOSE'] = previous_applicationEncoding['NAME_CASH_LOAN_PURPOSE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_CASH_LOAN_PURPOSE'])
le.classes_
previous_applicationEncoding['PREV_NAME_CASH_LOAN_PURPOSE'] = le.transform(previous_applicationEncoding['NAME_CASH_LOAN_PURPOSE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_CASH_LOAN_PURPOSE', axis = 1)

previous_applicationEncoding['NAME_CONTRACT_STATUS'] = previous_applicationEncoding['NAME_CONTRACT_STATUS'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_CONTRACT_STATUS'])
le.classes_
previous_applicationEncoding['PREV_NAME_CONTRACT_STATUS'] = le.transform(previous_applicationEncoding['NAME_CONTRACT_STATUS'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_CONTRACT_STATUS', axis = 1)

previous_applicationEncoding['NAME_PAYMENT_TYPE'] = previous_applicationEncoding['NAME_PAYMENT_TYPE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_PAYMENT_TYPE'])
le.classes_
previous_applicationEncoding['PREV_NAME_PAYMENT_TYPE'] = le.transform(previous_applicationEncoding['NAME_PAYMENT_TYPE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_PAYMENT_TYPE', axis = 1)

previous_applicationEncoding['CODE_REJECT_REASON'] = previous_applicationEncoding['CODE_REJECT_REASON'].fillna("Unknown")
le.fit(previous_applicationEncoding['CODE_REJECT_REASON'])
le.classes_
previous_applicationEncoding['PREV_CODE_REJECT_REASON'] = le.transform(previous_applicationEncoding['CODE_REJECT_REASON'])
previous_applicationEncoding = previous_applicationEncoding.drop('CODE_REJECT_REASON', axis = 1)

previous_applicationEncoding['NAME_TYPE_SUITE'] = previous_applicationEncoding['NAME_TYPE_SUITE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_TYPE_SUITE'])
le.classes_
previous_applicationEncoding['PREV_NAME_TYPE_SUITE'] = le.transform(previous_applicationEncoding['NAME_TYPE_SUITE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_TYPE_SUITE', axis = 1)

previous_applicationEncoding['NAME_CLIENT_TYPE'] = previous_applicationEncoding['NAME_CLIENT_TYPE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_CLIENT_TYPE'])
le.classes_
previous_applicationEncoding['PREV_NAME_CLIENT_TYPE'] = le.transform(previous_applicationEncoding['NAME_CLIENT_TYPE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_CLIENT_TYPE', axis = 1)

previous_applicationEncoding['NAME_GOODS_CATEGORY'] = previous_applicationEncoding['NAME_GOODS_CATEGORY'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_GOODS_CATEGORY'])
le.classes_
previous_applicationEncoding['PREV_NAME_GOODS_CATEGORY'] = le.transform(previous_applicationEncoding['NAME_GOODS_CATEGORY'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_GOODS_CATEGORY', axis = 1)

previous_applicationEncoding['NAME_PORTFOLIO'] = previous_applicationEncoding['NAME_PORTFOLIO'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_PORTFOLIO'])
le.classes_
previous_applicationEncoding['PREV_NAME_PORTFOLIO'] = le.transform(previous_applicationEncoding['NAME_PORTFOLIO'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_PORTFOLIO', axis = 1)

previous_applicationEncoding['NAME_PRODUCT_TYPE'] = previous_applicationEncoding['NAME_PRODUCT_TYPE'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_PRODUCT_TYPE'])
le.classes_
previous_applicationEncoding['PREV_NAME_PRODUCT_TYPE'] = le.transform(previous_applicationEncoding['NAME_PRODUCT_TYPE'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_PRODUCT_TYPE', axis = 1)

previous_applicationEncoding['CHANNEL_TYPE'] = previous_applicationEncoding['CHANNEL_TYPE'].fillna("Unknown")
le.fit(previous_applicationEncoding['CHANNEL_TYPE'])
le.classes_
previous_applicationEncoding['PREV_CHANNEL_TYPE'] = le.transform(previous_applicationEncoding['CHANNEL_TYPE'])
previous_applicationEncoding = previous_applicationEncoding.drop('CHANNEL_TYPE', axis = 1)

previous_applicationEncoding['NAME_YIELD_GROUP'] = previous_applicationEncoding['NAME_YIELD_GROUP'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_YIELD_GROUP'])
le.classes_
previous_applicationEncoding['PREV_NAME_YIELD_GROUP'] = le.transform(previous_applicationEncoding['NAME_YIELD_GROUP'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_YIELD_GROUP', axis = 1)

previous_applicationEncoding['PRODUCT_COMBINATION'] = previous_applicationEncoding['PRODUCT_COMBINATION'].fillna("Unknown")
le.fit(previous_applicationEncoding['PRODUCT_COMBINATION'])
le.classes_
previous_applicationEncoding['PREV_PRODUCT_COMBINATION'] = le.transform(previous_applicationEncoding['PRODUCT_COMBINATION'])
previous_applicationEncoding = previous_applicationEncoding.drop('PRODUCT_COMBINATION', axis = 1)

previous_applicationEncoding['NAME_SELLER_INDUSTRY'] = previous_applicationEncoding['NAME_SELLER_INDUSTRY'].fillna("Unknown")
le.fit(previous_applicationEncoding['NAME_SELLER_INDUSTRY'])
le.classes_
previous_applicationEncoding['PREV_NAME_SELLER_INDUSTRY'] = le.transform(previous_applicationEncoding['NAME_SELLER_INDUSTRY'])
previous_applicationEncoding = previous_applicationEncoding.drop('NAME_SELLER_INDUSTRY', axis = 1)

previous_applicationEncoding_summarise = previous_applicationEncoding.groupby(['SK_ID_CURR']).mean().reset_index()
bureauCount = bureau[['SK_ID_CURR','SK_ID_BUREAU']].groupby('SK_ID_CURR').count().reset_index()
prevCount = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count().reset_index()
prevCount.head()
trainData = trainCopy.merge(bureau_summarise, on = 'SK_ID_CURR', how = "left")
trainData = trainData.merge(credit_card_bal_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(installment_payments_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(pos_cash_bal_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(bureau_bal_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(previous_application_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(bureau_encoding_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(bureau_bal_encoding_summaries, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(credit_card_bal_encoding_summaries, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(pos_cash_bal_encoding_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(previous_applicationEncoding_summarise, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(bureauCount, on = 'SK_ID_CURR', how = 'left')
trainData = trainData.merge(prevCount, on = 'SK_ID_CURR', how = 'left')
trainData.head()
testCopy = test.copy()
traintestCopy = train.copy()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

testCopy['NAME_CONTRACT_TYPE'] = testCopy['NAME_CONTRACT_TYPE'].fillna("Unknown")
traintestCopy['NAME_CONTRACT_TYPE'] = traintestCopy['NAME_CONTRACT_TYPE'].fillna("Unknown")
le.fit(traintestCopy['NAME_CONTRACT_TYPE'])
le.classes_
testCopy['NAME_CONTRACT_TYPE'] = le.transform(testCopy['NAME_CONTRACT_TYPE'])


testCopy['CODE_GENDER'] = testCopy['CODE_GENDER'].fillna("Unknown")
traintestCopy['CODE_GENDER'] = traintestCopy['CODE_GENDER'].fillna("Unknown")
le.fit(traintestCopy['CODE_GENDER'])
le.classes_
testCopy['CODE_GENDER'] = le.transform(testCopy['CODE_GENDER'])


testCopy['FLAG_OWN_CAR'] = testCopy['FLAG_OWN_CAR'].fillna("Unknown")
traintestCopy['FLAG_OWN_CAR'] = traintestCopy['FLAG_OWN_CAR'].fillna("Unknown")
le.fit(traintestCopy['FLAG_OWN_CAR'])
le.classes_
testCopy['FLAG_OWN_CAR'] = le.transform(testCopy['FLAG_OWN_CAR'])


testCopy['FLAG_OWN_REALTY'] = testCopy['FLAG_OWN_REALTY'].fillna("Unknown")
traintestCopy['FLAG_OWN_REALTY'] = traintestCopy['FLAG_OWN_REALTY'].fillna("Unknown")
le.fit(traintestCopy['FLAG_OWN_REALTY'])
le.classes_
testCopy['FLAG_OWN_REALTY'] = le.transform(testCopy['FLAG_OWN_REALTY'])


testCopy['NAME_TYPE_SUITE'] = testCopy['NAME_TYPE_SUITE'].fillna("Unknown")
traintestCopy['NAME_TYPE_SUITE'] = traintestCopy['NAME_TYPE_SUITE'].fillna("Unknown")
le.fit(traintestCopy['NAME_TYPE_SUITE'])
le.classes_
testCopy['NAME_TYPE_SUITE'] = le.transform(testCopy['NAME_TYPE_SUITE'])


testCopy['NAME_INCOME_TYPE'] = testCopy['NAME_INCOME_TYPE'].fillna("Unknown")
traintestCopy['NAME_INCOME_TYPE'] = traintestCopy['NAME_INCOME_TYPE'].fillna("Unknown")
le.fit(traintestCopy['NAME_INCOME_TYPE'])
le.classes_
testCopy['NAME_INCOME_TYPE'] = le.transform(testCopy['NAME_INCOME_TYPE'])


testCopy['NAME_EDUCATION_TYPE'] = testCopy['NAME_EDUCATION_TYPE'].fillna("Unknown")
traintestCopy['NAME_EDUCATION_TYPE'] = traintestCopy['NAME_EDUCATION_TYPE'].fillna("Unknown")
le.fit(traintestCopy['NAME_EDUCATION_TYPE'])
le.classes_
testCopy['NAME_EDUCATION_TYPE'] = le.transform(testCopy['NAME_EDUCATION_TYPE'])


testCopy['NAME_FAMILY_STATUS'] = testCopy['NAME_FAMILY_STATUS'].fillna("Unknown")
traintestCopy['NAME_FAMILY_STATUS'] = traintestCopy['NAME_FAMILY_STATUS'].fillna("Unknown")
le.fit(traintestCopy['NAME_FAMILY_STATUS'])
le.classes_
testCopy['NAME_FAMILY_STATUS'] = le.transform(testCopy['NAME_FAMILY_STATUS'])


testCopy['NAME_HOUSING_TYPE'] = testCopy['NAME_HOUSING_TYPE'].fillna("Unknown")
traintestCopy['NAME_HOUSING_TYPE'] = traintestCopy['NAME_HOUSING_TYPE'].fillna("Unknown")
le.fit(traintestCopy['NAME_HOUSING_TYPE'])
le.classes_
testCopy['NAME_HOUSING_TYPE'] = le.transform(testCopy['NAME_HOUSING_TYPE'])


testCopy['OCCUPATION_TYPE'] = testCopy['OCCUPATION_TYPE'].fillna("Unknown")
traintestCopy['OCCUPATION_TYPE'] = traintestCopy['OCCUPATION_TYPE'].fillna("Unknown")
le.fit(traintestCopy['OCCUPATION_TYPE'])
le.classes_
testCopy['OCCUPATION_TYPE'] = le.transform(testCopy['OCCUPATION_TYPE'])


testCopy['ORGANIZATION_TYPE'] = testCopy['ORGANIZATION_TYPE'].fillna("Unknown")
traintestCopy['ORGANIZATION_TYPE'] = traintestCopy['ORGANIZATION_TYPE'].fillna("Unknown")
le.fit(traintestCopy['ORGANIZATION_TYPE'])
le.classes_
testCopy['ORGANIZATION_TYPE'] = le.transform(testCopy['ORGANIZATION_TYPE'])


testCopy['WEEKDAY_APPR_PROCESS_START'] = testCopy['WEEKDAY_APPR_PROCESS_START'].fillna("Unknown")
traintestCopy['WEEKDAY_APPR_PROCESS_START'] = traintestCopy['WEEKDAY_APPR_PROCESS_START'].fillna("Unknown")
le.fit(traintestCopy['WEEKDAY_APPR_PROCESS_START'])
le.classes_
testCopy['WEEKDAY_APPR_PROCESS_START'] = le.transform(testCopy['WEEKDAY_APPR_PROCESS_START'])


testCopy['FONDKAPREMONT_MODE'] = testCopy['FONDKAPREMONT_MODE'].fillna("Unknown")
traintestCopy['FONDKAPREMONT_MODE'] = traintestCopy['FONDKAPREMONT_MODE'].fillna("Unknown")
le.fit(traintestCopy['FONDKAPREMONT_MODE'])
le.classes_
testCopy['FONDKAPREMONT_MODE'] = le.transform(testCopy['FONDKAPREMONT_MODE'])


testCopy['HOUSETYPE_MODE'] = testCopy['HOUSETYPE_MODE'].fillna("Unknown")
traintestCopy['HOUSETYPE_MODE'] = traintestCopy['HOUSETYPE_MODE'].fillna("Unknown")
le.fit(traintestCopy['HOUSETYPE_MODE'])
le.classes_
testCopy['HOUSETYPE_MODE'] = le.transform(testCopy['HOUSETYPE_MODE'])


testCopy['WALLSMATERIAL_MODE'] = testCopy['WALLSMATERIAL_MODE'].fillna("Unknown")
traintestCopy['WALLSMATERIAL_MODE'] = traintestCopy['WALLSMATERIAL_MODE'].fillna("Unknown")
le.fit(traintestCopy['WALLSMATERIAL_MODE'])
le.classes_
testCopy['WALLSMATERIAL_MODE'] = le.transform(testCopy['WALLSMATERIAL_MODE'])


testCopy['EMERGENCYSTATE_MODE'] = testCopy['EMERGENCYSTATE_MODE'].fillna("Unknown")
traintestCopy['EMERGENCYSTATE_MODE'] = traintestCopy['EMERGENCYSTATE_MODE'].fillna("Unknown")
le.fit(traintestCopy['EMERGENCYSTATE_MODE'])
le.classes_
testCopy['EMERGENCYSTATE_MODE'] = le.transform(testCopy['EMERGENCYSTATE_MODE'])
testData = testCopy.merge(bureau_summarise, on = 'SK_ID_CURR', how = "left")
testData = testData.merge(credit_card_bal_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(installment_payments_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(pos_cash_bal_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(bureau_bal_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(previous_application_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(bureau_encoding_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(bureau_bal_encoding_summaries, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(credit_card_bal_encoding_summaries, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(pos_cash_bal_encoding_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(previous_applicationEncoding_summarise, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(bureauCount, on = 'SK_ID_CURR', how = 'left')
testData = testData.merge(prevCount, on = 'SK_ID_CURR', how = 'left')
testData.head()
del train, trainCopy, bureau, bureau_bal, bureau_bal_encoding, bureau_bal_encoding_summaries, bureau_bal_summarise, trainData
del bureau_balCopy, bureau_encoding, bureau_encoding_summarise, bureau_summarise
del credit_card_bal, credit_card_bal_encoding, credit_card_bal_encoding_summaries, credit_card_bal_summarise, credit_card_balCopy
del installment_payments, installment_payments_summarise, installment_paymentsCopy
del previous_application, previous_application_summarise, previous_applicationCopy, previous_applicationEncoding, previous_applicationEncoding_summarise
del pos_cash_bal, pos_cash_bal_encoding, pos_cash_bal_encoding_summarise, pos_cash_bal_summarise, pos_cash_balCopy
del test, testCopy, testData
#from guppy import hpy; h=hpy()
#h.heap()
