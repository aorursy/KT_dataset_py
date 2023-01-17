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

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
%matplotlib inline
df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
df_events.head()

df_app_events = pd.read_csv("../input/app_events.csv")
df_app_events.head()
df_app_labels = pd.read_csv("../input/app_labels.csv")
df_app_labels.head()
df_label_categories = pd.read_csv("../input/label_categories.csv")
df_label_categories.head()
df_phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv")
df_phone_brand_device_model.head()
df_new = pd.concat([df_phone_brand_device_model, df_events])
df_new
