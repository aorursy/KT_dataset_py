import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport 
from pycaret.classification import *
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
df = pd.read_csv('../input/churn_data_st.csv')
profile = ProfileReport(df,title='Pandas Profiling Report')
profile
print(df.shape)
df.head()
df_setup = setup(df, target='Churn',ignore_features=['customerID'])
compare_models(fold=5)
xgb_model=create_model('xgboost')
tuned_xgb_model=tune_model('xgboost', optimize = 'AUC')
# creating a decision tree model
dt = create_model('dt')# ensembling a trained dt model
dt_bagged = ensemble_model(dt)
plot_model(tuned_xgb_model, plot = 'auc')
plot_model(tuned_xgb_model, plot = 'pr')
plot_model(tuned_xgb_model,plot= 'boundary')
plot_model(tuned_xgb_model, plot='feature')
plot_model(tuned_xgb_model, plot = 'confusion_matrix')
predict_model(tuned_xgb_model)
xgb_final = finalize_model(tuned_xgb_model)
xgb_model
interpret_model(tuned_xgb_model)
interpret_model(tuned_xgb_model, plot= 'correlation')
predict_model(xgb_final)
save_model(tuned_xgb_model,'xgb_model')
ls
load_xgb=load_model('xgb_model')
load_xgb
evaluate_model(tuned_xgb_model)

from pycaret.datasets import get_data
df = get_data('diabetes')

from pycaret.classification import *
clf = setup(data= df, target ='Class variable')

lr = create_model('lr')

plot_model(lr) # other parameter plot='auc'/'boundary'/'pr'/'vc'
# Importing dataset
from pycaret.datasets import get_data
boston = get_data('boston')
# Importing module and initializing setup
from pycaret.regression import *
reg1 = setup(data = boston, target = 'medv')
# creating a model
lr = create_model('lr')
# plotting a model
plot_model(lr)
# Importing dataset
from pycaret.datasets import get_data
jewellery = get_data('jewellery')
# Importing module and initializing setup
from pycaret.clustering import *
clu1 = setup(data = jewellery)
# creating a model
kmeans = create_model('kmeans')
# plotting a model
plot_model(kmeans)

# Importing dataset
from pycaret.datasets import get_data
anomalies = get_data('anomaly')
# Importing module and initializing setup
from pycaret.anomaly import *
ano1 = setup(data = anomalies)
# creating a model
iforest = create_model('iforest')
# plotting a model
plot_model(iforest)



