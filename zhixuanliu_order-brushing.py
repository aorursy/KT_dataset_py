import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm 



csv_dir = "/kaggle/input/order-brushing-shopee-code-league/order_brush_order.csv"
csv = pd.read_csv( csv_dir )

csv.event_time = pd.to_datetime(csv.event_time)
def concern_rate(df): 

    time_list = df.event_time

    brush_users = []

    # identify shop with order brushing 

    for i in range(len(time_list)): 

        if True: 

            start = time_list.iloc[i]

            end = time_list.iloc[i] + pd.DateOffset(hours=1)  # 1 hour later 

            hr_df = df[ (df.event_time >= start) & (df.event_time <= end)]

            len_trade = len(hr_df)

            len_user = len(hr_df.userid.unique() )

            #print (len_trade, len_user)

            c_rate = len_trade/len_user

            if c_rate >=3: 

                cheat_sort = hr_df.userid.value_counts()

                cheat_user = cheat_sort[cheat_sort>=3].index

                for i_user in cheat_user: 

                    brush_users.append( str(i_user) ) 

            else: 

                pass 

        if True: 

            start = time_list.iloc[i] + pd.DateOffset(hours=-1) # 1 hour before 

            end = time_list.iloc[i]   

            hr_df = df[ (df.event_time >= start) & (df.event_time <= end)]

            len_trade = len(hr_df)

            len_user = len(hr_df.userid.unique() )

            #print (len_trade, len_user)

            c_rate = len_trade/len_user

            if c_rate >=3: 

                cheat_sort = hr_df.userid.value_counts()

                cheat_user = cheat_sort[cheat_sort>=3].index

                for i_user in cheat_user: 

                    brush_users.append( str(i_user) ) 

            else: 

                pass 

    if brush_users != []: 

        return "".join(list(set(brush_users)) )  # get unique users 

    else: 

        return "0"

result = []

unique_shop =  csv.shopid.unique()

for i_shop in (unique_shop): 

    shop_csv = csv[csv.shopid == i_shop].sort_values(by=['event_time'])

    brush_user = concern_rate(shop_csv)

    result.append(brush_user) 

    #print (brush_user)
shop_csv
concern_rate(shop_csv)
np.sum([i!='0' for i in result]) # number for order brushing 
df_ans = pd.DataFrame({'shopid': unique_shop, 'userid': result})

df_ans.to_csv('/kaggle/working/prediction.csv',index=False)