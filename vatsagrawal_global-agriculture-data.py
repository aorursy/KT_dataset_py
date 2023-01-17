# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
crop_data = pd.read_csv('../input/fao_data_crops_data.csv')
crop_data.head()
crop_data['country_or_area'].nunique()
india_data = crop_data[crop_data['country_or_area'] == 'India']
india_data.head()
india_crop_area = india_data[['year','value','category']]
india_crop_area.head()
india_crop_area = india_crop_area.rename(columns={'value' : 'area in Ha', 'category' : 'crop category code'})
india_crop_area.head()
india_crop_area.hist('area in Ha');
india_crop_area['crop category code'].nunique()
india_crop_area['crop category code'].nunique()
#path = r'C:\Users\Vatsal Agrawal\Desktop\india_crops.csv'
#india_crop_area.to_csv(path)
#print('done')
#this code snippet is not working. Ignore but do not delete.
#press (Shift+Enter) to run and display dataframe
india_crop_area.index.rename('Serial Number', inplace=True)
india_crop_area.sort_values(by = 'year',axis=0, ascending=False, inplace=True)
india_crop_area.reset_index(drop=True, inplace=True)
#india_crop_area.drop('Serial Number', axis=1, inplace=True)
india_crop_area
