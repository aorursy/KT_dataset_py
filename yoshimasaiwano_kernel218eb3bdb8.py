!pip install pycaret
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from pycaret.regression import *

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# loading other csvfile
df_loading = pd.read_csv("/kaggle/input/trends-assessment-prediction/loading.csv")
df_train_score = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")
# merge
df = pd.merge(df_train_score, df_loading, on = 'Id').dropna()
test_df = df_loading[~df_loading['Id'].isin(df_train_score['Id'])]  # ~はisin()に含まれないものを抽出

target_colms = list(df_train_score)
target_colms.pop(0)
def get_train_data(target):
    others = [tar for tar in target_colms if tar != target]
    train_df = df.drop(others, axis = 1)
    return train_df
models = []
def tune_and_ensemble(target):
    train_df = get_train_data(target)
    exp_reg = setup(
        data = train_df,
        target = target,
        train_size = 0.8,
        silent = True
    )
    
    tuned_model = tune_model('ridge')
    model = ensemble_model(tuned_model)
    return model
for i in range(5):
    model = tune_and_ensemble(target_colms[i])
    models.append(model)
for i in range(5):
    pred = predict_model(models[i], data = test_df)
    test_df[target_colms[i]] = pred['Label'].values
sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars = ["Id"], value_name = "Predicted")
sub_df["Id"] = sub_df["Id"].astype("str") + "_" + sub_df["variable"].astype("str")

sub_df = sub_df.drop("variable", axis = 1).sort_values("Id")

sub_df.to_csv("submission1.csv", index = False)
