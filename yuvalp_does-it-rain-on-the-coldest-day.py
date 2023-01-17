# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/austin_weather.csv")
df.Date.head()
from collections import Counter
rains = Counter(df['PrecipitationSumInches'])
rains.most_common(10)
sum(rains.values())
sum([v for r,v in rains.items() if r != "T" and float(r) > 0.1])
from datetime import datetime
df['Year'] = pd.Series([datetime.strptime(d,'%Y-%M-%d').year for d in df.Date])
df['Year'].head()
Counter(df['Year']).most_common()
min_temp_years = {y:df[df.Year == y].TempHighF.min() for y in [2013, 2014, 2015, 2016, 2017]}
min_temp_years
for rec in df.to_records():
    if rec.Year in min_temp_years and rec.TempHighF == min_temp_years[rec.Year]:
        print(rec.Date, rec.TempHighF, rec.PrecipitationSumInches)