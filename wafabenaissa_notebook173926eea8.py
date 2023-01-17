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
import csv

from collections import OrderedDict

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



with open('../input/documents_meta.csv', 'r') as f:

    r = csv.reader(f)

    dict2 = {row[0]: row[1:] for row in r}



with open('../input/promoted_content.csv', 'r') as f:

    r = csv.reader(f)

    dict1 = OrderedDict((row[0], row[1:]) for row in r)



result = OrderedDict()

for d in (dict1, dict2):

    for key, value in d.items():

        result.setdefault(key, []).extend(value)



with open('ab_combined.csv', 'w') as f:

    w = csv.writer(f)

    for key, value in result.items():

        w.writerow([key] + value)

        

data=pd.read_csv('ab_combined.csv',low_memory=False)

data