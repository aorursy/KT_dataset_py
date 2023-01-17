!pip install pycaret
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import regression module 

from pycaret.regression import * 



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
dataset.head()
dataset.info()
dataset.describe()
dataset.columns
set_config('seed', 999)
adm_reg = setup(dataset, target = 'Chance of Admit ')
best_models = compare_models(sort='RMSE')
ridge_model = create_model('ridge')
ridge_tuned = tune_model(ridge_model)
evaluate_model(ridge_model)
ridge_predict = predict_model(ridge_model)
lar = create_model('lar')

ridge = create_model('ridge')

blender = blend_models(estimator_list = [lar,ridge])
evaluate_model(blender)
predict_model(blender)