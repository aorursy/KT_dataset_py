# Basic libraries
import pandas as pd
import numpy as np
import os

# Plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# SKlearn related libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, zero_one_loss, hamming_loss

# Boosting technique algorithm
import xgboost as xgb

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Dataset path
DATA_PATH = "/kaggle/input/wineuci/Wine.csv"

columns = ['class','alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium',
    'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue',
    'od280/od315_of_diluted_wines', 'proline']

wine_data = pd.read_csv(DATA_PATH, names=columns, header=0)

wine_data.info()
print("=="*40)
wine_data.head()
# transform the label into 0, 1, 2
def trans_class(class_label):
    return int(class_label) - 1
wine_data['class'] = wine_data['class'].apply(trans_class)
np.unique(wine_data['class'])
wine_data.describe().T
print(f"Is there any null values : {wine_data.isnull().sum().any()}")
sns.set(style='whitegrid', palette='muted')

# Pairplot to see the attribute comparison
fig, ax = plt.subplots(1,2,figsize=(12,6))

sns.distplot(wine_data['alcohol'], kde=True, hist=True, ax=ax[0])

sns.distplot(wine_data['malic_acid'], kde=True, hist=True, ax=ax[1])

plt.show()

g = sns.jointplot(x=wine_data['alcohol'], y=wine_data['malic_acid'], color='r')

wine_data['class'].value_counts().plot.bar()
# Making X and Y data from the dataset
X = wine_data.loc[:, wine_data.columns != 'class'].values
y = wine_data['class'].values
print(f"Train shape : {X.shape}, Label : { y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train shape : {X_train.shape}, Label : { y_train.shape}")
dm_train = xgb.DMatrix(data=X_train, label=y_train)
dm_test = xgb.DMatrix(data=X_test)
# Key parameters
params = {
    'max_depth': 6,
    'min_child_weight':1,
    'objective': 'multi:softmax',
    'subsample':1,
    'colsample_bytree':1,
    'num_class': 3,
    'n_gpus': 0
}
xgb_clf = xgb.train(params, dm_train) # Train the model with dataset

# Prediction
predictions = xgb_clf.predict(dm_test)
predictions
print("Classification Report \n {}".format(classification_report(y_test, predictions)))
print("Misclassification rate {}".format(zero_one_loss(y_test, predictions, normalize=False)))
print("Hamming rate {:.2f}".format(hamming_loss(y_test, predictions)))