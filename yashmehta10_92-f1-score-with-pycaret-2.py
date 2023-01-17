#!pip install pycaret==2.0

# Importing six and using it instead of 'sklearn.externals.six' from imbalanced-learn
# Use this method when using fix_imbalance=True
import six
import sys
sys.modules['sklearn.externals.six'] = six

from pycaret.classification import *
from pycaret.utils import version
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# check version
from pycaret.utils import version
version()
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.shape
data.pop('Time')
data.head()

train_data, test_data = train_test_split(data, test_size=0.3)
test_labels = test_data.pop('Class') 
print(train_data.shape)
print(test_data.shape)
print(test_labels.shape)
class_counts = pd.value_counts(train_data['Class'], sort = True)
class_counts.plot(kind = 'bar', rot=0)
plt.title('Counts of Fraud/Normal')
plt.xticks(range(2), ['Not Fraud', 'Fraud'])
plt.xlabel("Class")
plt.ylabel("Count")
corr_matrix = train_data.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(corr_matrix[(corr_matrix >= 0.1) | (corr_matrix <= -0.1)], square=True, cmap='viridis', annot=True, linewidths=0.1)
fraud_detection = setup(data = train_data, target = 'Class', 
                   normalize = True,
                   transformation = True, transformation_method = 'yeo-johnson', 
                   fix_imbalance = True)
models()
compare_models(whitelist=['svm', 'ada' ,'mlp', 'dt', 'rf', 'et'])
rf = create_model('rf')
mlp = create_model('mlp')
xt = create_model('et')
rf = tune_model(rf)
mlp = tune_model(mlp)
xt = tune_model(xt)
blender = blend_models(estimator_list = [rf, mlp, xt])
model = finalize_model(blender)
predictions = predict_model(model, data = test_data)
from sklearn.metrics import f1_score, recall_score
f1 = f1_score(test_labels, predictions['Label'], average='macro')
recall = recall_score(test_labels, predictions['Label'], average='macro')
print("F1 Score= ",f1)
print("Recall= ", recall)