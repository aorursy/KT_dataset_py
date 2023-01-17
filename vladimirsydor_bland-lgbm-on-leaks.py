import pandas as pd

import numpy as np
fold_1 = pd.read_csv('/kaggle/input/lgbt-on-leaks-fold-1/submission.csv')

fold_2 = pd.read_csv('/kaggle/input/lgbt-on-leaks-fold-2/submission.csv')

fold_3 = pd.read_csv('/kaggle/input/lgbt-on-leaks-fold-3/submission.csv')

fold_4 = pd.read_csv('/kaggle/input/lgbt-on-leaks-fold-4/submission.csv')

fold_5 = pd.read_csv('/kaggle/input/lgbt-on-leaks-fold-5/submission.csv')
fold_1.head()
fold_1['meter_reading'] = (fold_1['meter_reading'] + fold_2['meter_reading'] + fold_3['meter_reading'] + fold_4['meter_reading'] + fold_5['meter_reading'])/5
fold_1.head()
fold_1['meter_reading'] = np.expm1(fold_1['meter_reading'])

fold_1.loc[fold_1['meter_reading'] < 0, 'meter_reading'] = 0
fold_1.head()
fold_1.to_csv('submission_fold_bland.csv',index=False)