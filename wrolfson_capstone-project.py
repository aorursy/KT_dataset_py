# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import statistics
%matplotlib inline
acci = pd.read_csv('../input/fars2017nationalcsv/accident.csv', usecols=[0, 1,2,5,9,10, 11, 12, 13,14,15,16,17, 25, 26, 50, 51])
acci.info()
