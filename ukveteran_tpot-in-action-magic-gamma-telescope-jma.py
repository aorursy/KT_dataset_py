# Import required libraries

from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split

import pandas as pd 

import numpy as np
#Load the data

telescope=pd.read_csv('../input/magic-gamma-telescope-data/MAGIC Gamma Telescope Data.csv')

telescope.head(5)
telescope_shuffle=telescope.iloc[np.random.permutation(len(telescope))]

telescope_shuffle.head()
tele=telescope_shuffle.reset_index(drop=True)

tele.head()
# Check the Data Type

tele.dtypes
for cat in ['Class']:

    print("Levels for catgeory '{0}': {1}".format(cat, tele[cat].unique()))
tele['Class']=tele['Class'].map({'g':0,'h':1})
tele = tele.fillna(-999)

pd.isnull(tele).any()
tele.shape
tele_class = tele['Class'].values
training_indices, validation_indices = training_indices, testing_indices = train_test_split(tele.index, stratify = tele_class, train_size=0.75, test_size=0.25)

training_indices.size, validation_indices.size
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=15)

tpot.fit(tele.drop('Class',axis=1).loc[training_indices].values, tele.loc[training_indices,'Class'].values)
tpot.score(tele.drop('Class',axis=1).loc[validation_indices].values, tele.loc[validation_indices, 'Class'].values)
tpot.export('tpot_MAGIC_Gamma_Telescope_pipeline.py')