!pip install pycaret
import pandas as pd
data_classification = pd.read_csv('../input/loan-prediction/train_loan.csv')
data_classification.head()
from pycaret import classification
classification_setup = classification.setup(data= data_classification, target='Loan_Status')
classification_dt = classification.create_model('dt')
classification_xgb = classification.create_model('xgboost')
boosting = classification.ensemble_model(classification_dt, method= 'Boosting')
blender = classification.blend_models(estimator_list=[classification_dt, classification_xgb])
classification.compare_models()
# AUC-ROC plot
classification.plot_model(classification_dt, plot = 'auc')
# Decision Boundary
classification.plot_model(classification_dt, plot = 'boundary')
# Precision Recall Curve
classification.plot_model(classification_dt, plot = 'pr')
# Validation Curve
classification.plot_model(classification_dt, plot = 'vc')
classification.evaluate_model(classification_dt)
classification.interpret_model(classification_xgb)
classification.interpret_model(classification_xgb,plot='correlation')
test_data_classification = pd.read_csv('../input/loan-prediction/test_loan.csv')
predictions = classification.predict_model(classification_dt, data=test_data_classification)
predictions
classification.save_model(classification_dt, 'decision_tree_1')
dt_model = classification.load_model(model_name='decision_tree_1')