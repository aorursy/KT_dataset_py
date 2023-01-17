# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

cereal = pd.read_csv('../input/cereal.csv')

cereal.head(3)

cereal['sodium'].describe()

cereal['sugars'].describe()

result = scipy.stats.ttest_ind(cereal['sodium'], cereal['sugars'], equal_var = False)

result

sns.distplot(cereal['sodium'], kde = False).set_title('Level of sodium in cereals')

sns.distplot(cereal['sugars'], kde = False).set_title('Level of sugar in cereals')
