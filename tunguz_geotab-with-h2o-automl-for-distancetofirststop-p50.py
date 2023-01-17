import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
print(os.listdir("../input/baseline-feature-engineering-geotab-69-5-lb"))

import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
train = h2o.import_file("../input/baseline-feature-engineering-geotab-69-5-lb/train_danFeatsV1.csv.gz")

test = h2o.import_file("../input/baseline-feature-engineering-geotab-69-5-lb/test_danFeatsV1.csv.gz")
x = ["IntersectionId",

             'Intersection',

           'diffHeading',  'same_street_exact',

           "Hour","Weekend","Month",

          'Latitude', 'Longitude',

          'EntryHeading', 'ExitHeading',

            'Atlanta', 'Boston', 'Chicago',

       'Philadelphia']



y = 'DistanceToFirstStop_p50'
aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=30000)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
# If you need to generate predictions on a test set, you can make

# predictions directly on the `"H2OAutoML"` object, or on the leader

# model object directly



preds = aml.predict(test)
preds = preds.as_data_frame().values.flatten()

preds
np.save('preds_'+y, preds)