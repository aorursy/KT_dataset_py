import pytz
import pandas as pd
import numpy as np
dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8'
    }
#only read 100,000 rows for illustration
train = pd.read_csv('../input/train.csv', dtype = dtypes, nrows = 100000)
train.head(5)
cst = pytz.timezone('Asia/Shanghai')
train.click_time = pd.to_datetime(train.click_time, errors = 'ignore')
train['local_click_time'] = train['click_time'].dt.tz_localize(pytz.utc).dt.tz_convert(cst)
train.head(5)