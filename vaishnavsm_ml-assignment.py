import numpy as np
import pandas as pd

# Encoders
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Plotting
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

from collections import namedtuple

from tqdm import tqdm
# !ls /kaggle/input
# raw_csv = pd.read_csv('/kaggle/input/uci-car-evaluation-data-set/car.data', names=["Buying Price", "Mainteinence Price", "Doors", "Persons", "Luggage Boot", "Safety", "Target"])
raw_csv = pd.read_csv('/kaggle/input/ucisatimage/sat.trn', sep=" ", header=None)
raw_csv.head()
# Only for satellite dataset
data_csv = raw_csv.copy()
# Encode Values - Only for CARS dataset, abandoned due to high accuracy of 97%+
# data_csv = pd.DataFrame()
# label_encoder = LabelEncoder()
# data_csv["Target"] = label_encoder.fit_transform(raw_csv["Target"])
# print("Unique Labels: ", data_csv["Target"].unique())

# Use OneHot Encoders for X-values to not add the connotations of a sliding scale between labels
# def onehot_enc(df, key, concat=None):
#     data = df[[key]]
#     ohe = OneHotEncoder()
#     td = pd.DataFrame(ohe.fit_transform(data).toarray())
#     td.columns = [f"{key}_{ohe.categories_[0][col]}" for col in td.columns]
#     if concat is not None:
#         concat = concat.join(td)
#         return ohe, concat
#     df = df.join(td)
#     return ohe, df

# buy_enc, data_csv = onehot_enc(raw_csv, "Buying Price", data_csv)
# maintain_enc, data_csv = onehot_enc(raw_csv, "Mainteinence Price", data_csv)
# doors_enc, data_csv = onehot_enc(raw_csv, "Doors", data_csv)
# persons_enc, data_csv = onehot_enc(raw_csv, "Persons", data_csv)
# lugboot_enc, data_csv = onehot_enc(raw_csv, "Luggage Boot", data_csv)
# safety_enc, data_csv = onehot_enc(raw_csv, "Safety", data_csv)

# data_csv.head()

# Visualize the classes
# raw_csv["Target"].hist()

raw_csv[36].hist()

# Unfortunately, it is quite skewed
# Y = np.array(data_csv["Target"])
# X = np.array(data_csv.drop(["Target"], axis=1))

Y = np.array(data_csv[36])
X = np.array(data_csv.drop([36], axis=1))

x_train, x_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.30,
                                                    random_state=0,
                                                    stratify=Y)

x_train.shape, x_test.shape, y_train.shape, y_test.shape
Hyperparameter = namedtuple('Hyperparameters', 'learning_rate n_hidden l2_alpha momentum')

NUM_PER_PARAM = 20

params = [
    Hyperparameter(learning_rate = 0.01, n_hidden = 5, l2_alpha = 0.0001, momentum=0.9), # Baseline
    Hyperparameter(learning_rate = 0.001, n_hidden = 5, l2_alpha = 0.0001, momentum=0.9), # Change lr
    Hyperparameter(learning_rate = 0.1, n_hidden = 5, l2_alpha = 0.0001, momentum=0.9), # Change lr
    Hyperparameter(learning_rate = 0.01, n_hidden = 50, l2_alpha = 0.0001, momentum=0.9), # Change n_hidden
    Hyperparameter(learning_rate = 0.01, n_hidden = 100, l2_alpha = 0.0001, momentum=0.9), # Change n_hidden
    Hyperparameter(learning_rate = 0.01, n_hidden = 5, l2_alpha = 0.00001, momentum=0.9), # Change l2_alpha
    Hyperparameter(learning_rate = 0.01, n_hidden = 5, l2_alpha = 0.001, momentum=0.9), # Change l2_alpha
    Hyperparameter(learning_rate = 0.01, n_hidden = 5, l2_alpha = 0.0001, momentum=0.5), # Change momentum
    Hyperparameter(learning_rate = 0.01, n_hidden = 5, l2_alpha = 0.0001, momentum=0.1), # Change momentus
    Hyperparameter(learning_rate = 0.001, n_hidden = 100, l2_alpha = 0.0001, momentum=0.9), # One test from experience
]
%timeit 
results = []

scoring = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "recall": "recall_macro"
}

with tqdm(total=NUM_PER_PARAM*len(params)) as tq:
    for param in params:
        results.append([])
        for iter in range(NUM_PER_PARAM):
            clf = MLPClassifier(
                tol=1e-5, 
                learning_rate="constant", 
                learning_rate_init=param.learning_rate, 
                hidden_layer_sizes=(param.n_hidden,),
                activation="logistic",
                solver="sgd",
                alpha=param.l2_alpha,
                momentum=param.momentum,
                max_iter=10000000000
               )
            scores = cross_validate(clf, x_train, y_train, cv=3, scoring=scoring)
            results[-1].append(scores)
            tq.update()
accuracy_results = pd.DataFrame()
for i, result in enumerate(results):
    res = []
    for r in result:
        res.extend(r['test_acc'])
    accuracy_results[f"Parameter {i}"]=pd.Series(np.array(sorted(res)))
sns.boxplot(data=accuracy_results)
f1_results = pd.DataFrame()
for i, result in enumerate(results):
    res = []
    for r in result:
        res.extend(r['test_f1'])
    f1_results[f"Parameter {i}"]=pd.Series(np.array(sorted(res)))
sns.boxplot(data=f1_results)
params = [
    Hyperparameter(learning_rate = 0.001, n_hidden = 100, l2_alpha = 0.0001, momentum=0.9), # Baseline (Best from last run)
    Hyperparameter(learning_rate = 0.001, n_hidden = 400, l2_alpha = 0.00001, momentum=0.9),
    Hyperparameter(learning_rate = 0.001, n_hidden = 200, l2_alpha = 0.00001, momentum=0.9),
    Hyperparameter(learning_rate = 0.0001, n_hidden = 100, l2_alpha = 0.00001, momentum=0.9),
    Hyperparameter(learning_rate = 0.0001, n_hidden = 100, l2_alpha = 0.000001, momentum=0.7),
    Hyperparameter(learning_rate = 0.0001, n_hidden = 100, l2_alpha = 0.00001, momentum=0.7),
]
%timeit 
results = []

scoring = {
    "acc": "accuracy",
    "f1": "f1_macro",
    "recall": "recall_macro"
}

with tqdm(total=NUM_PER_PARAM*len(params)) as tq:
    for param in params:
        results.append([])
        for iter in range(NUM_PER_PARAM):
            clf = MLPClassifier(
                tol=1e-5, 
                learning_rate="constant", 
                learning_rate_init=param.learning_rate, 
                hidden_layer_sizes=(param.n_hidden,),
                activation="logistic",
                solver="sgd",
                alpha=param.l2_alpha,
                momentum=param.momentum,
                max_iter=10000000000
               )
            scores = cross_validate(clf, x_train, y_train, cv=3, scoring=scoring)
            results[-1].append(scores)
            tq.update()
accuracy_results = pd.DataFrame()
for i, result in enumerate(results):
    res = []
    for r in result:
        res.extend(r['test_acc'])
    accuracy_results[f"Parameter {i}"]=pd.Series(np.array(sorted(res)))
sns.boxplot(data=accuracy_results)
# Final Model
param = Hyperparameter(learning_rate = 0.0001, n_hidden = 400, l2_alpha = 0.00001, momentum=0.9)
clf = MLPClassifier(
                tol=1e-5, 
                learning_rate="constant", 
                learning_rate_init=param.learning_rate, 
                hidden_layer_sizes=(param.n_hidden,),
                activation="logistic",
                solver="sgd",
                alpha=param.l2_alpha,
                momentum=param.momentum,
                max_iter=10000000000
               )

clf.fit(x_train, y_train)
y_pred_train = clf.predict(x_train)
print(classification_report(y_train, y_pred_train))
y_pred_test = clf.predict(x_test)
print(classification_report(y_test, y_pred_test))
def predict(estimator, x, rank=1):
    predict_probs = estimator.predict_proba(x)
    return estimator.classes_[predict_probs.argsort()[:,::-1][:,:rank]]


def cmc(estimator, x, y_true, up_to_rank=10000):
    result = []
    ranks = min(up_to_rank, len(estimator.classes_))
    full_res = predict(estimator, x, ranks)
    for rank in range(1, ranks+1):
        result.append(np.average([y_true[i] in full_res[i, :rank] for i in range(y_true.shape[0])]))
    return result
        
cmcc = cmc(clf, x_train, y_train)
plt.figure()
plt.plot([i+1 for i in range(len(clf.classes_))], cmcc)
plt.title("Cumulative Match Characteristic Curve")
plt.show()
