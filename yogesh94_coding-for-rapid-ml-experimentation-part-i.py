def example_fn(param_one, param_two):

    ''' 
    This is an example function showing how to write
    docstrings

    Parameters
    ----------
    param_one: pd.DataFrame
               Description of the param_one argument
    
    param_two: int
               Description of the param_two argument

    Returns
    -------
    return_data: pd.DataFrame
                 An example description of the returned      
                 object

    Suggested Imports
    ----------------
    import numpy as np
    import pandas as pd

    Example Usage
    -------------
    transformed_data = example_func(param_one, param_two)

    '''
    
    pass
help(example_fn)
from typing import List

def list_squared(input_list: List[int]) -> List[int]:
    return [element**2 for element in input_list]

list_squared([2, 4, 6, 8])
import numpy as np
import pandas as pd
train_data = pd.read_csv('../input/rossmann-store-sales/train.csv', low_memory=False)
train_data.head()
def generate_date_features(input_data: pd.DataFrame,
                           date_col:str,
                           use_col_name: bool=True,
                           inplace: bool=False) -> pd.DataFrame:
    
    ''' 
    generates date features from a date column, such as,
    year, month, day, etc. which can be used to train
    Machine Learning models. 

    Parameters
    ----------
    input_data : pd.DataFrame
                 The input data frame
          
    date_col : str
               The column name of the date column for which
               the features have to be generated
               
               Ensure that the column is a pandas datetime
               object. And also has year, month and day in
               it.
               
    use_col_name : bool, default True
                   If True, the column name will be appended
                   to the name of the feature that has been
                   created
    
    inplace : bool, default False
              If False, a new data frame object is returned.
              Else, the same data frame passed as input is
              modified

    Returns
    -------
    return_data: pd.DataFrame
                 The returned dataframe which has the appended
                 date based features

    Suggested Imports
    ----------------
    import numpy as np
    import pandas as pd

    Example Usage
    -------------
    data_with_date_features = generate_date_features(data, 'date')

    '''
    
    # Inplace or Copy
    if inplace:
        data_frame = input_data
    else:
        data_frame = input_data.copy()
        
    # Use column name
    if use_col_name:
        new_col_name = f'{date_col}_'
    else:
        new_col_name = ''
        
    # ensure that the column is converted to a 
    # pandas datetime object 
    data_frame[date_col] = pd.to_datetime(data_frame[date_col])
    
    # Generate date features
    data_frame[f'{new_col_name}year'] = data_frame[date_col].dt.year
    data_frame[f'{new_col_name}month'] = data_frame[date_col].dt.month
    data_frame[f'{new_col_name}day'] = data_frame[date_col].dt.day
    data_frame[f'{new_col_name}weeknum'] = data_frame[date_col].dt.weekofyear
    data_frame[f'{new_col_name}dayofweek'] = data_frame[date_col].dt.dayofweek
    data_frame[f'{new_col_name}quarter'] = data_frame[date_col].dt.quarter
    
    return data_frame
new_features = generate_date_features(train_data, date_col='Date', use_col_name=False)
new_features.head()
from typing import Optional, List

def create_time_kfolds_days(input_data: pd.DataFrame,
                            date_col: str, num_days: int,
                            group_cols: Optional[List[str]] = None,
                            num_folds: int = 5, inplace: bool = False
                           ) -> pd.DataFrame:
    ''' 
    Creates a kfold column based on time, which allows us to validate our
    machine learning models. This works for when the forecasts are at
    the day level.

    Parameters
    ----------
    input_data : pd.DataFrame
                 The input data frame
           
    date_col : str
               The column name of the date column for which
               the features have to be generated
               
               Ensure that the column is a pandas datetime
               object. And also has year, month and day in
               it.
          
    num_days: int
              The number of days that have to be included
              in each validation fold
               
    group_cols : List[str] or None, default None
                 If a list of strings is passed, the folds
                 are created for the last num_days by grouping
                 the columns passed.
                 
                 If None, the creates the folds by just using
                 the date_col
                 
    num_folds: int default 5
               The number of folds to create
    
    inplace : bool, default False
              If False, a new data frame object is returned.
              Else, the same data frame passed as input is
              modified

    Returns
    -------
    return_data: pd.DataFrame
                 The returned dataframe which has the kfold
                 column appended to it.

    Suggested Imports
    ----------------
    import numpy as np
    import pandas as pd
    from typing import Optional, List

    Example Usage
    -------------
    1) Accessing training and validation datasets for a particular fold number
    
    data_with_kfold = create_time_based_folds(data, 'date', 28, group_cols=['store', 'item'])
    train_for_fold_1 = data_with_kfold[data_with_kfold['kfold'] != 1]
    val_for_fold_1 = data_with_kfold[data_with_kfold['kfold'] == 1]

    '''
    
    # Inplace or Copy
    if inplace:
        data_frame = input_data
    else:
        data_frame = input_data.copy()
        
    # ensure that the column is converted to a 
    # pandas datetime object 
    data_frame[date_col] = pd.to_datetime(data_frame[date_col])
    
    # Sort all observations by date for each of
    # the groupby columns
    data_frame = data_frame.groupby(group_cols)\
                 .apply(lambda x: x.sort_values(by=date_col))\
                 .reset_index(drop=True)
    
    max_date_in_data = data_frame[date_col].max()

    date_ranges = []
    
    for idx, fold_num in enumerate(range(1, num_folds+1)):
        date_range_fold = pd.date_range(max_date -\
                                        pd.DateOffset(fold_num*(num_days-1)),\
                                        max_date -\
                                        pd.DateOffset((fold_num-1)*(num_days-1))
                                       )
        date_ranges.append(date_range_fold)
    
    data_frame['kfold'] = -1
    
    date_folds = zip(reversed(date_ranges), range(num_folds))
    for idx, (date_range, fold_num) in enumerate(date_folds):
        data_frame.loc[data_frame[date_col].isin(date_range), 'kfold'] = fold_num
            
    return data_frame
kfold_data = create_time_kfolds_days(train_data, 'Date', 48, ['Store'])
train_kfold = kfold_data[kfold_data.kfold < 4]

val_kfold = kfold_data[kfold_data.kfold == 4]
train_kfold.Date.min()
train_kfold.Date.max()
val_kfold.Date.min()
val_kfold.Date.max()