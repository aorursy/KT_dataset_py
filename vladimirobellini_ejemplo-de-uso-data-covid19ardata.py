# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import date, timedelta

# carga de csv en pandas
df = pd.read_csv(os.path.join(dirname, filename))

# print de dataframe
df

# datos de ayer
today = date.today()
yesterday = date.today() - timedelta(days=1)
yesterday = yesterday.strftime("%d/%m/%Y")

dftoday = df[(df['fecha'] == yesterday) & (df['osm_admin_level_4'] != 'Indeterminado')]
dftoday

