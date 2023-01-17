



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"

train = pd.read_csv(train_url)



print(train)
