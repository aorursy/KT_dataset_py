dEnd = 1941 #last day for which weights are calculated; 1913 -> weights for validation period; 1941 -> weights for evaluation period
import numpy as np

import pandas as pd
sales = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_evaluation.csv") 

selling_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv") 

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")
### drop unnecessary 'd_xxx' columns in sales dataframe

sales = sales.drop(columns=[f'd_{ii:d}' for ii in range(1,dEnd-28+1)] + [f'd_{ii:d}' for ii in range(dEnd+1,1942)])

sales.head(3)
# create a daily sell price dataframe to multiply it with sales to get total revenues

calendar['d'] = calendar['d'].str[2:].astype(int)

selling_prices.set_index(pd.Index(selling_prices['item_id'] + '_' + selling_prices['store_id'] + '_validation', name='id'), inplace=True)

daily_prices = selling_prices.join(calendar[['wm_yr_wk','d']].set_index('wm_yr_wk').d, on='wm_yr_wk')

daily_prices = daily_prices[(daily_prices['d']>dEnd-28) & (daily_prices['d']<=dEnd)] #drop unnecessary 'd's

daily_prices = daily_prices.pivot_table(values='sell_price', index='id', columns='d').reset_index()

daily_prices.head(3)
daily_prices = daily_prices.sort_values('id') #ensure same order in daily_prices and sales for multiplication

sales = sales.sort_values('id')

sales.iloc[:,6:] = sales.iloc[:,6:].values * daily_prices.iloc[:,1:].values #total revenues

sales.head(3)
### build aggragations and calculate weights; Note: Although still called 'sales' the dataframe now contains total revenue

sales['total'] = 'total' #-> groupby 'total' creates one group with all elements

group_ids = ( 'total', 'state_id', 'store_id', 'cat_id', 'dept_id', 

        ['state_id', 'cat_id'],  ['state_id', 'dept_id'], ['store_id', 'cat_id'],

        ['store_id', 'dept_id'], 'item_id', ['state_id','item_id'], ['item_id', 'store_id'])

Weight = []

Agg_Level_1 = []

Agg_Level_2 = []

Level = []

for level in range(len(group_ids)):

    weight_level = []

    tmp = sales.groupby(group_ids[level]).sum()

    for idx, row in tmp.iterrows(): #tqdm(tmp.iterrows()):

        if type(idx) != tuple:

            idx = (idx,'X')

        Agg_Level_1 += [idx[0]]

        Agg_Level_2 += [idx[1]]

        Level += ['Level'+str(level+1)]

        Weight += [np.sum(row.values)]

Weight = np.array(Weight)/Weight[0] #normalization by total revenue

weights = pd.DataFrame({'Level_id':Level,'Agg_Level_1':Agg_Level_1, 'Agg_Level_2':Agg_Level_2, 'Weight':Weight})
weights.to_csv('weights.csv')
weights.head(3)
### if computed weights for the validation period, compare to weights provided by the organizer on github (available as dataset on kaggle)

if dEnd == 1913:

    weights_val = pd.read_csv("/kaggle/input/m5methods/validation/weights_validation.csv") 



    weights.sort_values(['Level_id','Agg_Level_1','Agg_Level_2'],inplace=True) #ensure same ordering of weight dataframes for comparison

    weights_val.sort_values(['Level_id','Agg_Level_1','Agg_Level_2'],inplace=True)

    print(np.any(np.abs(weights['Weight'].values - weights_val['Weight'].values)>1e-7))