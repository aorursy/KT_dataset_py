import itertools

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import svm, model_selection, metrics
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.drop(columns=['Id'], inplace=True)

df['Species'] = df['Species'].astype('category')

df.head()
df['Species'].unique()
df.describe()
_ = pd.plotting.scatter_matrix(df, figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)

_ = plt.title('Correlations')
df.groupby('Species').count().plot.pie(y='SepalLengthCm', figsize=(8, 8))

plt.ylabel('')

_ = plt.title('Ratio of species')
df['SpeciesCat'] = df['Species'].cat.codes

df.sample(5)
for column in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:

    df[f'{column}Norm'] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
feature_columns = ['SepalLengthCmNorm', 'SepalWidthCmNorm', 'PetalLengthCmNorm', 'PetalWidthCmNorm']

features = df[['SepalLengthCmNorm', 'SepalWidthCmNorm', 'PetalLengthCmNorm', 'PetalWidthCmNorm']]

target = df['SpeciesCat']



features_train, features_test, target_train, target_test = model_selection.train_test_split(

    features, target, test_size=0.25, random_state=0, shuffle=True)
classifier = svm.SVC(kernel='linear', probability=True, random_state=0)

classifier.fit(features_train, target_train)
pred_prob = classifier.predict_proba(features_train)

pred = classifier.predict(features_train)



plt.figure(figsize=(16, 5))

for label in target_train.unique():

    fpr, tpr, _ = metrics.roc_curve(target_train, pred_prob[:, label], pos_label=label)

    f1 = metrics.f1_score(target_train, pred, average='micro', labels=[label])

    ax = plt.plot(fpr, tpr, label=f'Class: {label}, F1 {f1 :.3f}',  alpha=0.5, marker='.')

auc = metrics.roc_auc_score(target_train, pred_prob, multi_class='ovr', average='macro', labels=[0, 1, 2])

plt.title(f'ROC AUC curves for each label (AUC score: {auc :.3f}) - training dataset')

plt.xlabel('FPR')

plt.ylabel('TPR')

plt.legend()

plt.show()
results = []

results.append({

    'dataset': 'training',

    'balanced_accuracy_score': metrics.balanced_accuracy_score(target_train, pred),

    'cohen_kappa_score': metrics.cohen_kappa_score(target_train, pred),

    'matthews_corrcoef':  metrics.matthews_corrcoef(target_train, pred),

})



pd.DataFrame(results)
plt.figure(figsize=(8, 6))

sns.heatmap(metrics.confusion_matrix(target_train, pred), annot=True)

_ = plt.title('Confusion matrix for predictions for training dataset')
feature_combinations = list(itertools.combinations(feature_columns, r=2))

_, axs = plt.subplots(2, len(feature_combinations)//2, figsize=(36, 12))



misclassification_indexes = []

for i, true_and_pred in enumerate(zip(target_train, pred)):

    sample, sample_prediction = true_and_pred

    if sample != sample_prediction:

        misclassification_indexes.append(i)



flatten = lambda l: [item for sublist in l for item in sublist]

axs = flatten(axs)

for i, f in enumerate(feature_combinations):

    f1, f2 = f

    scatter = axs[i].scatter(features_train[f1], features_train[f2], c=pred)



    for misclass_idx in misclassification_indexes:

        scatter = axs[i].scatter(features_train[f1].iloc[misclass_idx], features_train[f2].iloc[misclass_idx], c='red', s=[50])

    axs[i].set_title(f'Classification resutls on {f1} vs {f2} for training dataset')

    axs[i].set_xlabel(f1)

    axs[i].set_ylabel(f2)

    axs[i].legend(['correct prediction', 'misclassification'])
pred_prob_test = classifier.predict_proba(features_test)

pred_test = classifier.predict(features_test)



plt.figure(figsize=(16, 5))

for label in target_train.unique():

    fpr, tpr, _ = metrics.roc_curve(target_test, pred_prob_test[:, label], pos_label=label)

    f1 = metrics.f1_score(target_test, pred_test, average='micro', labels=[label])

    ax = plt.plot(fpr, tpr, label=f'Class: {label}, F1 {f1 :.3f}',  alpha=0.5, marker='.')

auc = metrics.roc_auc_score(target_test, pred_prob_test, multi_class='ovr', average='macro', labels=[0, 1, 2])



plt.title(f'ROC AUC curves for each label (AUC score: {auc :.3f}) - training dataset')

plt.xlabel('FPR')

plt.ylabel('TPR')

_ = plt.legend()
results.append({

    'dataset': 'test',

    'balanced_accuracy_score': metrics.balanced_accuracy_score(target_test, pred_test),

    'cohen_kappa_score': metrics.cohen_kappa_score(target_test, pred_test),

    'matthews_corrcoef':  metrics.matthews_corrcoef(target_test, pred_test),

})



df_results = pd.DataFrame(results)

df_results
plt.figure(figsize=(8, 6))

sns.heatmap(metrics.confusion_matrix(target_test, pred_test), annot=True)

_ = plt.title('Confusion matrix for predictions for testing dataset')
feature_combinations = list(itertools.combinations(feature_columns, r=2))

_, axs = plt.subplots(2, len(feature_combinations)//2, figsize=(36, 12))



misclassification_indexes = []

for i, true_and_pred in enumerate(zip(target_test, pred_test)):

    sample, sample_prediction = true_and_pred

    if sample != sample_prediction:

        misclassification_indexes.append(i)



flatten = lambda l: [item for sublist in l for item in sublist]

axs = flatten(axs)

for i, f in enumerate(feature_combinations):

    f1, f2 = f

    scatter = axs[i].scatter(features_test[f1], features_test[f2], c=pred_test)



    for misclass_idx in misclassification_indexes:

        scatter = axs[i].scatter(features_test[f1].iloc[misclass_idx], features_test[f2].iloc[misclass_idx], c='red', s=[50])

    axs[i].set_title(f'Classification resutls on {f1} vs {f2} for testing dataset')

    axs[i].set_xlabel(f1)

    axs[i].set_ylabel(f2)

    axs[i].legend(['correct prediction', 'misclassification'])