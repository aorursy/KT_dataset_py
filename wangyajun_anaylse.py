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
app_lables = pd.read_csv('../input/app_labels.csv')
app_events = pd.read_csv('../input/app_events.csv')
events = pd.read_csv('../input/events.csv')
gender_age_test = pd.read_csv('../input/gender_age_test.csv')
label_ctg = pd.read_csv('../input/label_categories.csv')
phone_bdm = pd.read_csv('../input/phone_brand_device_model.csv')
gender_age_train = pd.read_csv('../input/gender_age_train.csv')

df = app_events.groupby('event_id').app_id.count()
gender_age_train.head()
app_events.app_id.count()
app_events.app_id.nunique()