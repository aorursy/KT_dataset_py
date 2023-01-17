# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

pl = pd.read_csv("../input/seattle_pet_licenses.csv")
pl.license_issue_date = pl.license_issue_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%f'))
# print(pl.license_issue_date.map(lambda x: x.year))
ym = pl.isnull().animal_s_name.groupby([pl.license_issue_date.map(lambda x: x.month), pl.license_issue_date.map(lambda x: x.isoweekday())]).count()
ym.index.names = ['month', 'day of the week']
print(ym)
# Any results you write to the current directory are saved as output.

ym.unstack(level='month').plot.line()
