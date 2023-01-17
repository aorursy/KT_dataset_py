%pylab inline
from sklearn import ensemble, model_selection, metrics 



import numpy as np

import pandas as pd
bioresponce = pd.read_csv('..//input/train.csv', header=0, sep=',')

bioresponce_test = pd.read_csv('..//input/test.csv')
bioresponce.head()
bioresponce.shape
bioresponce.columns
bioresponce_target = bioresponce.Activity.values
print('bioresponse = 1: {:.2f}\nbioresponse = 0: {:.2f}'.format(sum(bioresponce_target)/float(len(bioresponce_target)), 

                1.0 - sum(bioresponce_target)/float(len(bioresponce_target))))
bioresponce_data = bioresponce.iloc[:, 1:]
rf_classifier_low_depth = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 2, random_state = 1)
train_sizes, train_scores, test_scores = model_selection.learning_curve(rf_classifier_low_depth, bioresponce_data, bioresponce_target, 

                                                                       train_sizes=np.arange(0.1,1., 0.2), 

                                                                       cv=3, scoring='accuracy')
print(train_sizes)

print(train_scores.mean(axis = 1))

print(test_scores.mean(axis = 1))
pylab.grid(True)

pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')

pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')

pylab.ylim((0.0, 1.05))

pylab.legend(loc='lower right')
rf_classifier_low_depth.fit(bioresponce_data, bioresponce_target)

result_low = rf_classifier_low_depth.predict(bioresponce_test)

df_low = pd.DataFrame(result_low)



df_low.index.name='MoleculeId'

df_low.index+=1

df_low.columns=['PredictedProbability']

#df_low.to_csv('results_low_tree.csv', header=True)

df_low.head()
rf_classifier = ensemble.RandomForestClassifier(n_estimators = 50, max_depth = 10, random_state = 1)
train_sizes, c, test_scores = model_selection.learning_curve(rf_classifier, bioresponce_data, bioresponce_target, 

                                                                       train_sizes=np.arange(0.1,1, 0.2), 

                                                                       cv=3, scoring='accuracy')
pylab.grid(True)

pylab.plot(train_sizes, train_scores.mean(axis = 1), 'g-', marker='o', label='train')

pylab.plot(train_sizes, test_scores.mean(axis = 1), 'r-', marker='o', label='test')

pylab.ylim((0.0, 1.05))

pylab.legend(loc='lower right')
rf_classifier.fit(bioresponce_data, bioresponce_target)
result = rf_classifier.predict(bioresponce_test)

df = pd.DataFrame(result)
df.index.name='MoleculeId'

df.index+=1

df.columns=['PredictedProbability']

df.to_csv('results_without_proba.csv', header=True)

df.head()