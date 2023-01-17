



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.

train_data = pd.read_csv("../input/train.csv")

train_data.shape

test_data = pd.read_csv("../input/test.csv")

test_data.shape
train_data.head()
test_data.shape

test_data.head()