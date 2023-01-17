from fastai.tabular import *

import os, tarfile

import random

import matplotlib.pyplot as plt

import pandas as pd

import re

from datetime import *



%matplotlib inline

%reload_ext autoreload

%autoreload 2



np.random.seed(23)

np.set_printoptions(threshold=50, edgeitems=20)
!ls -al /kaggle/input/rossmann-time-series-data-engineering
OUTPUT = '/kaggle/working/'

PATH='/kaggle/input/rossmann-time-series-data-engineering/'

df = pd.read_feather(f'{PATH}df')

train_df = pd.read_feather(f'{PATH}joined2')

test_df = pd.read_feather(f'{PATH}joined2_test')

train_df.shape, test_df.shape
cat_vars = ['Store', 'DayOfWeek', 'Promo',

       'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Week', 'Day',

       'Is_year_end', 'Is_year_start', 'StoreType', 'Assortment', 

       'Promo2', 'PromoInterval', 'State',   

       'Events',  'CompetitionMonthsOpen', 

       'Promo2Weeks',

       'SchoolHoliday_bw','StateHoliday_bw', 'Promo_bw', 'SchoolHoliday_fw', 'StateHoliday_fw','Promo_fw', 

       'SchoolHoliday_DaySum', 'StateHoliday_DaySum', 'Promo_DaySum', 

       'SchoolHoliday_DayCount', 'StateHoliday_DayCount', 'Promo_DayCount']



cont_vars = ['CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',

            'Max_Humidity','Mean_Humidity', 'Min_Humidity',

            'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h','Precipitationmm','CloudCover',

            'trend', 'trend_DE', 'CompetitionDaysOpen', 'Promo2Days',

            'AfterSchoolHoliday', 'BeforeSchoolHoliday', 'AfterStateHoliday',

            'BeforeStateHoliday', 'AfterPromo', 'BeforePromo']



dep_var = 'Sales'

df = train_df[cat_vars + cont_vars + [dep_var,'Date']].copy()
test_df['Date'].min(), test_df['Date'].max(), len(test_df)
cut = train_df['Date'][(train_df['Date'] == train_df['Date'][len(test_df)])].index.max()

valid_idx = range(cut) ; valid_idx
train_df['Date'][0], train_df['Date'][cut] 
procs=[FillMissing, Categorify, Normalize]



datalist = (TabularList.from_df(df, path=OUTPUT, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)

                .split_by_idx(valid_idx=valid_idx)

                .label_from_df(cols=dep_var, label_cls=FloatList, log=True)

                .add_test(TabularList.from_df(test_df, path=PATH, cat_names=cat_vars, cont_names=cont_vars)))

data = datalist.databunch(bs=512)
defaults.device
max_log_y = np.log(np.max(train_df['Sales'])*1.2)  # whether it is better to have +20% max sales need to be verified

y_range = torch.tensor([0, max_log_y], device=defaults.device)

learn = tabular_learner(data, layers=[1000,500], ps=[0.001,0.01], emb_drop=0.04, 

                        y_range=y_range, metrics=exp_rmspe)
learn.to_fp16
learn.data.batch_size

import fastai

fastai.__version__
learn.lr_find(end_lr=100, wd=0.3)

learn.recorder.plot()
learn.fit_one_cycle(5, 3e-3, wd=0.3)
learn.save('bs512_5ep_2e-3_wd0.3')
learn.fit_one_cycle(5, 1e-3, wd=0.3)
learn.recorder.plot_losses()
learn.save('bs512_2_5ep_1e-3_wd0.3')
learn.fit_one_cycle(5, 3e-4, wd=0.3)

learn.recorder.plot_losses()
learn.save('bs512_3_5ep_3e-4_wd0.3')
data = datalist.databunch(bs=128)

learn.data = data

learn.data.batch_size
learn.fit_one_cycle(5, 1e-3, wd=0.2)

learn.recorder.plot_losses()
learn.save('bs128_5ep_1e-3_wd0.2')
learn.fit_one_cycle(5, 1e-3, wd=0.2)

learn.recorder.plot_losses()
learn.fit_one_cycle(5, 1e-3, wd=0.1)

learn.recorder.plot_losses()
learn.fit_one_cycle(20, 1e-3, wd=0.1)

learn.recorder.plot_losses()
# learn.fit_one_cycle(5, 5e-4, wd=0.1)

# learn.recorder.plot_losses()


learn.save('last')
test_preds=learn.get_preds(DatasetType.Test)

test_df["Sales"]=np.exp(test_preds[0].data).numpy().T[0]

test_df[["Id","Sales"]]=test_df[["Id","Sales"]].astype("int")

test_df[["Id","Sales"]].to_csv("rossmann_submission.csv",index=False)