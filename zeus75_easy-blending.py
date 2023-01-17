#Load libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Upload submission

sub=pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
#Dataset loaded for blending 

#https://www.kaggle.com/duykhanh99/lgb-fe-0-9492-lb-newfeature-0-9496-lb

#https://www.kaggle.com/duykhanh99/lightgbm-feature-engineering-eda-with-r

#https://www.kaggle.com/kyakovlev/ieee-gb-2-make-amount-useful-again

#https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419

#https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm

#https://www.kaggle.com/kyakovlev/ieee-lgbm-with-groupkfold-cv

#https://www.kaggle.com/tolgahancepel/lightgbm-single-model-and-feature-engineering

#https://www.kaggle.com/rafay12/is-it-really-fraud

#https://www.kaggle.com/ysjf13/cis-fraud-detection-visualize-feature-engineering

#https://www.kaggle.com/whitebird/a-method-to-valid-offline-lb-9506

#https://www.kaggle.com/kyakovlev/ieee-simple-lgbm
df0=pd.read_csv('../input/a-method-to-valid-offline-lb-9506/simple_ensemble6.csv')

df1=pd.read_csv('../input/lgb-fe-0-9492-lb-newfeature-0-9496-lb/submission.csv')

df2=pd.read_csv('../input/lightgbm-feature-engineering-eda-with-r/submission.csv')

df3=pd.read_csv('../input/ieee-gb-2-make-amount-useful-again/submission.csv')

df4=pd.read_csv('../input/lgb-single-model-lb-0-9419/ieee_cis_fraud_detection_v2.csv')

df5=pd.read_csv('../input/feature-engineering-lightgbm/submission.csv')

df6=pd.read_csv('../input/ieee-lgbm-with-groupkfold-cv/submission.csv')

df7=pd.read_csv('../input/lightgbm-single-model-and-feature-engineering/submission.csv')

df8=pd.read_csv('../input/is-it-really-fraud/submission.csv')

df9=pd.read_csv('../input/cis-fraud-detection-visualize-feature-engineering/prediction.csv')

df10=pd.read_csv('../input/ieee-simple-lgbm/submission.csv')
blend1 = df0['isFraud']*0.7 + df1['isFraud']*0.06 + df2['isFraud']*0.06 + df3['isFraud']*0.06 + df4['isFraud']*0.06 + df5['isFraud']*0.06
sub['isFraud'] = blend1

df11 = sub
blend2 = df11['isFraud']*0.5 + df6['isFraud']*0.1 + df7['isFraud']*0.1 + df8['isFraud']*0.1 + df9['isFraud']*0.1 + df10['isFraud']*0.1
sub1 = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')
sub1['isFraud'] = blend2

sub1.head()
sub1.to_csv('easy_blend4.csv',index=False)