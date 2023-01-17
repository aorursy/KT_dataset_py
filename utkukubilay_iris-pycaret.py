
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)

iris = pd.read_csv("/kaggle/input/iris/Iris.csv") # the iris dataset is now a Pandas DataFrame


warnings.filterwarnings("ignore")
!pip install pycaret
iris.drop("Id",axis = 1, inplace = True)
iris.head()
from pycaret.classification import *
msk = np.random.rand(len(iris)) < 0.75
train = iris[msk]
test = iris[~msk]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
clf1 = setup(data = train, 
             target = "Species",
             silent = True,
             remove_outliers = True,
             feature_selection = True)
lgbm  = create_model('lightgbm') 
tuned_lightgbm = tune_model('lightgbm')
evaluate_model(tuned_lightgbm)
pred_lgbm = predict_model(tuned_lightgbm, data=test)
pred_lgbm['preds'] = pred_lgbm['Label']
pred_lgbm.head()
knn = create_model('knn')
tuned_knn = tune_model('knn')
plot_model(tuned_knn, plot = 'confusion_matrix')
plot_model(tuned_lightgbm, plot = 'confusion_matrix')
plot_model(tuned_knn, plot='boundary')
plot_model(tuned_knn, plot = 'error')
final_knn = finalize_model(tuned_knn)
final_lgbm = finalize_model(tuned_lightgbm)
predict_model(final_knn);
predict_model(final_lgbm);
plot_model(estimator = tuned_lightgbm, plot = 'learning')
plot_model(estimator = tuned_lightgbm, plot = 'auc')