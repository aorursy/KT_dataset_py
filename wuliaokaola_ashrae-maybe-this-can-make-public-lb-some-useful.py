import warnings

warnings.filterwarnings('ignore')

import os

import gc

import pickle

import numpy as np

import pandas as pd

import random as rn

import matplotlib.pyplot as plt



from tqdm import tqdm

from datetime import datetime, timedelta
test = pd.read_csv('../input/ashrae-energy-prediction/test.csv', index_col=0, parse_dates = ['timestamp'])

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv', usecols=['site_id', 'building_id'])
test = test.merge(building, left_on = "building_id", right_on = "building_id", how = "left")

t = test[['building_id', 'meter', 'timestamp']]

t['row_id'] = t.index
submission_base = pd.read_csv('../input/ashrae-half-and-half/submission.csv', index_col=0)
submission = submission_base.copy()
site_0 = pd.read_csv('../input/new-ucf-starter-kernel/submission_ucf_replaced.csv', index_col=0)

submission.loc[test[test['site_id']==0].index, 'meter_reading'] = site_0['meter_reading']

del site_0

gc.collect()
with open('../usr/lib/ucl_data_leakage_episode_2/site1.pkl', 'rb') as f:

    site_1 = pickle.load(f)

site_1 = site_1[site_1['timestamp'].dt.year > 2016]
site_1 = site_1.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")

site_1 = site_1[['meter_reading_scraped', 'row_id']].set_index('row_id').dropna()

submission.loc[site_1.index, 'meter_reading'] = site_1['meter_reading_scraped']

del site_1

gc.collect()
site_2 = pd.read_csv('../input/asu-buildings-energy-consumption/asu_2016-2018.csv', parse_dates = ['timestamp'])

site_2 = site_2[site_2['timestamp'].dt.year > 2016]
site_2 = site_2.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")

site_2 = site_2[['meter_reading', 'row_id']].set_index('row_id').dropna()

submission.loc[site_2.index, 'meter_reading'] = site_2['meter_reading']

del site_2

gc.collect()
site_4 = pd.read_csv('../input/ucb-data-leakage-site-4/site4.csv', parse_dates = ['timestamp'])

site_4.columns = ['building_id', 'timestamp', 'meter_reading']

site_4['meter'] = 0

site_4['timestamp'] = pd.DatetimeIndex(site_4['timestamp']) + timedelta(hours=-8)

site_4 = site_4[site_4['timestamp'].dt.year > 2016]
site_4 = site_4.merge(t, left_on = ['building_id', 'meter', 'timestamp'], right_on = ['building_id', 'meter', 'timestamp'], how = "left")

site_4 = site_4[['meter_reading', 'row_id']].dropna().set_index('row_id')

submission.loc[site_4.index, 'meter_reading'] = site_4['meter_reading']

del site_4

gc.collect()
submission.to_csv('submission.csv')
submission.describe()