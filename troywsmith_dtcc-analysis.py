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
# Load Data

csv_path = f'/kaggle/input/dtcc_commodities.csv'

df = pd.read_csv(csv_path, thousands=',')

pd.set_option('display.max_columns', 5000)
df_yesterday = df[(df['EXECUTION_TIMESTAMP'] > '2020-06-08T00:00:00') & (df['EXECUTION_TIMESTAMP'] < '2020-06-09T12:00:00')]
# Oil Volume (lots)

oil_swaps_yesterday = df_yesterday[df_yesterday['TAXONOMY'] == 'Commodity:Energy:Oil:Swap:Cash']

cols = ['EXECUTION_TIMESTAMP', 'UNDERLYING_ASSET_1', 'PRICE_NOTATION_TYPE', 'PRICE_NOTATION', 'ROUNDED_NOTIONAL_AMOUNT_1']

oil_swaps_yesterday_volume = oil_swaps_yesterday[cols]

oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"] = oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"].str.replace(',', '')

oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"] = oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"].str.replace('+', '')

oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"] = oil_swaps_yesterday_volume["ROUNDED_NOTIONAL_AMOUNT_1"].astype(int)

oil_swaps_yesterday_volume.head()
oil_swaps_yesterday_volume['Volume (lots)'] = oil_swaps_yesterday_volume.ROUNDED_NOTIONAL_AMOUNT_1 / oil_swaps_yesterday_volume.PRICE_NOTATION / 1000

oil_swaps_yesterday_volume.head()
oil_swaps_yesterday_volume.groupby(['UNDERLYING_ASSET_1']).sum()