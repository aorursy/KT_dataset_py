import pandas as pd
import itertools
from datetime import datetime
df = pd.read_csv('../input/order-brushing-dataset-shopee-code-league-week-1/order_brush_order.csv')
df.sort_values(['shopid', 'event_time'], ascending=[True, True], inplace=True)
shopid_grouped = df.groupby('shopid')
#transform time
fmt = '%Y-%m-%d %H:%M:%S'

def transform_time(ts):
    return datetime.strptime(str(ts), fmt)

# Returns time difference in minutes
def time_difference(ts1, ts2):
    time_diff = transform_time(ts2)-transform_time(ts1)
    return int(round(time_diff.total_seconds()/60))
# Takes in dataframe and returns array of suspicious user ids
def filter_suspicious_users(df):
    if len(df) <= 2:
        return []
    
    df = df.reset_index(drop=True)

    # Find row where the difference from the last row is 60 mins or less
    stop_row_index = 0
    last_row = df.iloc[-1]
    for index, row in df.iterrows():
        if time_difference(row['event_time'], last_row['event_time']) <= 60:
            stop_row_index = index
            break

    limit = stop_row_index + 1
    last_row_index = df.last_valid_index()
    
    # Interate through rows looking for suspicious users in 6o min window
    results = []
    for index, row in itertools.islice(df.iterrows(), limit):
        user_count = {}
        order_count = 1
        ts1 = row['event_time']
        if index < last_row_index:
                index+=1
        else:
            break
        user_count[row['userid']] = 1
        
        # Add all user ids and order counts within 60 mins window to user_count
        while (time_difference(ts1, df.iloc[index, 3]) <= 60 and index <= last_row_index):
            userid = df.iloc[index, 2]
            order_count+=1
            if userid in user_count:
                user_count[userid] = user_count[userid] + 1
            else:
                user_count[userid] = 1
            if index < last_row_index:
                index+=1
            else:
                break
        
        conc_rate = order_count/len(user_count)

        if conc_rate >= 3:
            highest = max(user_count.values())
            users = [k for k, v in user_count.items() if v == highest]
            results.append(users)
    results = [item for sublist in results for item in sublist]
    return results
results = []
for name, group in shopid_grouped:
    # Call filter_suspicious_users which returns array of users
    # Append shopid (name) and userids to results
    user_id_arr = filter_suspicious_users(group)
    if len(user_id_arr) > 1:
        user_id_arr.sort()
        userids = '&'.join(map(str, user_id_arr))
        results.append([name, userids])
    elif len(user_id_arr) == 1:
        results.append([name, user_id_arr[0]])
    else:
        results.append([name, 0])

results_df = pd.DataFrame(results, columns=['shopid', 'userid'])
results_df.head()