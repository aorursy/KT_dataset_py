import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
glass_data = pd.read_csv('/kaggle/input/glass/glass.csv')

# Glass data
glass_data.info()
print("--"*20)
glass_data.head()
glass_data.values[50:60, :]
glass_data['Type'].value_counts()
plt.figure(figsize=(10,6))
glass_data.hist()
plt.show()
target = glass_data.values[:, -1]
counter = Counter(target)

for k, v in counter.items():
    per = v / len(target) * 100
    print('Class=%d, Count=%d, Percentage=%.2f%%' % (k,v,per))
def load_dataset(file_path):
    df = pd.read_csv(file_path, header=0)
    
    data = df.values
    #Split data into input and output
    X, y = data[:, :-1], data[:, -1]
    # Encode the label data
    y = LabelEncoder().fit_transform(y)
    
    return X, y
    
# Evaluate the model
def evaluate_model(X, y, model):
    K = 5
    R = 3
    # K-Fold on the data
    cv = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=1)
    
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    
    return scores

def get_models():
    models, names = list(), list()
    # SVM
    models.append(SVC(gamma='auto'))
    names.append('SVC')
    
    models.append(KNeighborsClassifier())
    names.append('KNN')
    
    models.append(BaggingClassifier(n_estimators=1000))
    names.append('BAG')
    
    models.append(RandomForestClassifier(n_estimators=1000))
    names.append('RF')
    
    models.append(ExtraTreesClassifier(n_estimators=1000))
    names.append('ET')
    
    return models, names
file_path = '/kaggle/input/glass/glass.csv'

X, y = load_dataset(file_path)

models, names = get_models()

results = list()

for i in range(len(models)):
    scores = evaluate_model(X, y, models[i])
    results.append(scores)
    print('>%s %.3f (%.3f)' % (names[i], np.mean(scores), np.std(scores)))

plt.boxplot(results, labels=names, showmeans=True)
plt.show()
class_weights = {0:1.0, 1:1.0, 2:2.0, 3:2.0, 4:2.0, 5:2.0}

rf_model = RandomForestClassifier(n_estimators=1000, class_weight=class_weights)

et_model = ExtraTreesClassifier(n_estimators=1000, class_weight=class_weights)

#Evaluate model
scores = evaluate_model(X, y, rf_model)
et_score = evaluate_model(X, y, et_model)

print("RF Mean Accuracy: %.3f (%.3f)" % (np.mean(scores), np.std(scores)))
print("ET Mean Accuracy: %.3f (%.3f)" % (np.mean(et_score), np.std(et_score)))
rf_model.fit(X, y)

row = [ 1.5232,13.72, 3.72,0.51, 71.75,  0.09 ,10.06 ,  0.0,  0.16  ]

print('>Predicted=%d (expected 0)' % (rf_model.predict([row])))
