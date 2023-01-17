# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import h2o

print(h2o.__version__)

from h2o.automl import H2OAutoML



h2o.init(max_mem_size='16G')
%%time

train = h2o.import_file("/kaggle/input/trends-train-test-creator/train.csv")

features = np.load('/kaggle/input/trends-train-test-creator/features.npy')
x = list(features)

y = 'domain1_var2'

aml = H2OAutoML(max_models=500, seed=47, max_runtime_secs=31000)

aml.train(x=x, y=y, training_frame=train)
# View the AutoML Leaderboard

lb = aml.leaderboard

lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)
# The leader model is stored here

aml.leader
test = h2o.import_file("/kaggle/input/trends-train-test-creator/test.csv")
preds = aml.predict(test)

preds.head()
preds = preds.as_data_frame().values
np.save('preds_domain1_var2', preds)