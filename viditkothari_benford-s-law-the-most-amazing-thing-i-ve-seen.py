# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
path = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        path.append(os.path.join(dirname, filename))
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pop2k = pd.read_csv(path[0])
pop2k10 = pd.read_csv(path[1])
pop2k.columns
pop2k.head(2)
pop2k.isna().sum()
pop2k = pop2k[['population', 'zipcode']]
pop2k10 = pop2k10[['population', 'zipcode']]
for i in list(pop2k.columns):
    pop2k[i+'_firstdigit'] = pop2k[i].apply(lambda x: str(x)[:1])
for i in list(pop2k10.columns):
    pop2k10[i+'_firstdigit'] = pop2k10[i].apply(lambda x: str(x)[:1])
pop2k.head(2)
pop2k.population_firstdigit.value_counts(normalize = True)*100
pop2k10.population_firstdigit.value_counts(normalize = True)*100
pop2k.zipcode_firstdigit.value_counts(normalize = True)*100
pop2k10.zipcode_firstdigit.value_counts(normalize = True)*100