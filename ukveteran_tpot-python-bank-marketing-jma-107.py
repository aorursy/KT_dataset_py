from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

import pandas as pd 

import numpy as np
Marketing=pd.read_csv('../input/bank-marketing/bank-additional-full.csv',  sep = ';')

Marketing.head(5)
Marketing.groupby('loan').y.value_counts()
Marketing.groupby(['loan','marital']).y.value_counts()
Marketing.rename(columns={'y': 'class'}, inplace=True)
Marketing.dtypes
for cat in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome' ,'class']:

    print("Number of levels in category '{0}': \b {1:2.2f} ".format(cat, Marketing[cat].unique().size))
for cat in ['contact', 'poutcome','class', 'marital', 'default', 'housing', 'loan']:

    print("Levels for catgeory '{0}': {1}".format(cat, Marketing[cat].unique()))
Marketing['marital'] = Marketing['marital'].map({'married':0,'single':1,'divorced':2,'unknown':3})

Marketing['default'] = Marketing['default'].map({'no':0,'yes':1,'unknown':2})

Marketing['housing'] = Marketing['housing'].map({'no':0,'yes':1,'unknown':2})

Marketing['loan'] = Marketing['loan'].map({'no':0,'yes':1,'unknown':2})

Marketing['contact'] = Marketing['contact'].map({'telephone':0,'cellular':1})

Marketing['poutcome'] = Marketing['poutcome'].map({'nonexistent':0,'failure':1,'success':2})

Marketing['class'] = Marketing['class'].map({'no':0,'yes':1})
Marketing = Marketing.fillna(-999)

pd.isnull(Marketing).any()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()



job_Trans = mlb.fit_transform([{str(val)} for val in Marketing['job'].values])

education_Trans = mlb.fit_transform([{str(val)} for val in Marketing['education'].values])

month_Trans = mlb.fit_transform([{str(val)} for val in Marketing['month'].values])

day_of_week_Trans = mlb.fit_transform([{str(val)} for val in Marketing['day_of_week'].values])
day_of_week_Trans
marketing_new = Marketing.drop(['marital','default','housing','loan','contact','poutcome','class','job','education','month','day_of_week'], axis=1)
assert (len(Marketing['day_of_week'].unique()) == len(mlb.classes_)), "Not Equal" #check correct encoding done
Marketing['day_of_week'].unique(),mlb.classes_
marketing_new = np.hstack((marketing_new.values, job_Trans, education_Trans, month_Trans, day_of_week_Trans))
np.isnan(marketing_new).any()
marketing_new[0].size
marketing_class = Marketing['class'].values
training_indices, validation_indices = training_indices, testing_indices = train_test_split(Marketing.index, stratify = marketing_class, train_size=0.75, test_size=0.25)

training_indices.size, validation_indices.size
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=15)

tpot.fit(marketing_new[training_indices], marketing_class[training_indices])
tpot.score(marketing_new[validation_indices], Marketing.loc[validation_indices, 'class'].values)