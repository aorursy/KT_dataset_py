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
%ls ..
% ls ../input
PATH = "../input/"



metal_bands_df = pd.read_csv(PATH + 'metal_bands_2017.csv', encoding='latin1')

world_pop_df   = pd.read_csv(PATH + 'world_population_1960_2015.csv', encoding='latin-1')
metal_bands_df.describe()
world_pop_df.describe()