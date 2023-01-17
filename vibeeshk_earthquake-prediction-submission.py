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
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR

from sklearn.metrics import mean_absolute_error
train = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
# Create a training file with simple derived features



rows = 150_000

segments = int(np.floor(train.shape[0] / rows))



X_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['ave', 'std', 'max', 'min'])

y_train = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])



for segment in tqdm(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = seg['acoustic_data'].values

    y = seg['time_to_failure'].values[-1]

    

    y_train.loc[segment, 'time_to_failure'] = y

    

    X_train.loc[segment, 'ave'] = x.mean()

    X_train.loc[segment, 'std'] = x.std()

    X_train.loc[segment, 'max'] = x.max()

    X_train.loc[segment, 'min'] = x.min()
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



X_train_scaled
svm = NuSVR()

svm.fit(X_train_scaled, y_train.values.flatten())

y_pred = svm.predict(X_train_scaled)
submission = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=submission.index)

for seg_id in X_test.index:

    seg = pd.read_csv('/kaggle/input/LANL-Earthquake-Prediction/test/' + seg_id + '.csv')

    

    x = seg['acoustic_data'].values

    

    X_test.loc[seg_id, 'ave'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()
X_test_scaled = scaler.transform(X_test)

submission['time_to_failure'] = svm.predict(X_test_scaled)

submission.to_csv('submission.csv')