# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#let us read the data

raw_data = pd.read_csv("/kaggle/input/vg-sales-pred/Data/Train.csv")

test_data = pd.read_csv("/kaggle/input/vg-sales-pred/Data/Test.csv")
raw_data.head()
import catboost
#Keeping in the ID part was a crazy idea

CReg = catboost.CatBoostRegressor(cat_features=['CONSOLE', 'CATEGORY', 'PUBLISHER', 'RATING'], learning_rate=0.03)



CReg.fit(raw_data.drop(['SalesInMillions'], axis=1), 

         raw_data['SalesInMillions'], verbose=0)
print(CReg.best_score_)
sub_file = pd.read_csv("/kaggle/input/vg-sales-pred/Data/Sample_Submission.csv")
sub_file['SalesInMillions'] = CReg.predict(test_data)
# sub = pd.read_excel("/kaggle/input/mh-financial-risk/Financial_Risk_Participants_Data/Sample_Submission.xlsx")



# submit_df = pd.DataFrame(test_prediction,  columns=['0','1'])



from IPython.display import FileLink

sub_name = "cat_baseline3"

sub_file.to_csv(sub_name+'.csv', index=False)

FileLink(sub_name+'.csv')
print(pd.read_csv("/kaggle/input/vg-sales-pred/Data/Sample_Submission.csv").head())