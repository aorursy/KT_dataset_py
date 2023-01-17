!pip install pycaret
!pip install shap
import sys
!{sys.executable} -m pip install -U pandas-profiling[notebook]
!jupyter nbextension enable --py widgetsnbextension
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
from pandas_profiling import ProfileReport
from pycaret.classification import *
df = pd.read_csv("../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
profile = ProfileReport(df, title='Heart Failure Profiling Report', explorative=True)
profile.to_widgets()
set_config('seed', 999)
heart_failure_clf = setup(df, target='DEATH_EVENT',silent=True)
best = compare_models()
cat = create_model('catboost')
cat_bagged = ensemble_model(cat, method = 'Bagging')
interpret_model(cat)
heart_failure_pred = predict_model(cat)