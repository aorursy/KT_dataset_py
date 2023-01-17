import pandas as pd

import numpy as np
def reduce_mem_usage(df, verbose=True, downcast_float=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if downcast_float:

                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                        df[col] = df[col].astype(np.float16)

                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                        df[col] = df[col].astype(np.float32)

                    else:

                        df[col] = df[col].astype(np.float64)    

                

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
X_test = reduce_mem_usage(pd.read_parquet('/kaggle/input/baseline-preprocessing-leaks-train-fe/X_test.parquet.gzip'), downcast_float=True)
fold_1 = np.load('/kaggle/input/nn-on-pp-leaks-train-fe-fold-1/prediction.npy')

fold_2 = np.load('/kaggle/input/nn-on-pp-leaks-train-fe-fold-2/prediction.npy')

fold_3 = np.load('/kaggle/input/nn-on-pp-leaks-train-fe-fold-3/prediction.npy')

fold_4 = np.load('/kaggle/input/nn-on-pp-leaks-train-fe-fold-4/prediction.npy')

fold_5 = np.load('/kaggle/input/nn-on-pp-leaks-train-fe-fold-5/prediction.npy')
bland = pd.DataFrame({

    'row_id':np.array(X_test['row_id']),

    'meter_reading':(fold_1 + fold_2 + fold_3+ fold_4+ fold_5)/5

})
bland.head()
bland['meter_reading'] = np.expm1(bland['meter_reading'])

bland.loc[bland['meter_reading'] < 0, 'meter_reading'] = 0
bland.head()
bland.to_csv('submission_fold_bland.csv',index=False)