import pandas as pd

import numpy as np
def first_leak_pp(df):

    df.fillna(0, inplace=True)

    df.loc[df.meter_reading < 0, 'meter_reading'] = 0 # remove large negative values

    df = df[df.building_id!=245]

    df['meter'] = df['meter'].astype(int)

    return df



class LeakAgregator(object):

    def __init__(self, list_of_leak_dicts):

        self.list_of_leak_dicts = list_of_leak_dicts

        self.leak_dfs = [el['preprocessing'](el['read'](el['path'])) for el in self.list_of_leak_dicts]

        self.overall_df = pd.concat(self.leak_dfs, axis=0)
leak_dict = [

    {

        'path':'../input/ashrae-leak-data-station/leak.feather',

        'read':pd.read_feather,

        'preprocessing':first_leak_pp

    }

]
ob = LeakAgregator(leak_dict)
test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv', parse_dates=['timestamp'])

test = test.merge(ob.overall_df, on=['building_id','meter','timestamp'], how='left')
ob.overall_df.to_csv('leak_df.csv',index=False)

test[['row_id','meter_reading']].to_csv('leaked_test_target.csv',index=False)