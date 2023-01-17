import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_excel('../input/covid19/dataset.xlsx')
positive = df.loc[df['SARS-Cov-2 exam result'] == 'positive']
negative = df.loc[df['SARS-Cov-2 exam result'] == 'negative']
negative_99_missing_fields = set([col for col in negative.columns if negative[col].isnull().mean() >= 0.99])
positive_99_missing_fields = set([col for col in positive.columns if positive[col].isnull().mean() >= 0.99])
assuming_not_important = list(negative_99_missing_fields.intersection(positive_99_missing_fields))
assuming_not_important
exclude_col = ['Patient ID', 'SARS-Cov-2 exam result', 'Phosphor', 'Albumin', 'Patient age quantile', 
               'Ionized calcium', 'Urine - Density', 'Urine - Red blood cells', 'Ferritin', 'Ionized calcium\xa0']
categorical_field = [col for col in df.columns if df[col].dtype == 'object' and col not in assuming_not_important+exclude_col]
numerical_fields = [col for col in df.columns if col not in categorical_field+exclude_col]
continuous_fields = [col for col in numerical_fields if len(df[col].unique()) > 10]
fig, ax = plt.subplots(len(continuous_fields), 3, figsize=(13,200))
row = 0
for item in range(0, len(continuous_fields)):
    ax[row, 0].scatter(negative[continuous_fields[item]],negative[continuous_fields[item]], c='yellow')
    ax[row, 0].scatter(positive[continuous_fields[item]],positive[continuous_fields[item]], c='red')
    ax[row, 0].set_title(continuous_fields[item])
    ax[row, 0].grid(True)

    ax[row, 1].hist(negative[continuous_fields[item]].dropna(), color='yellow')
    ax[row, 1].hist(positive[continuous_fields[item]].dropna(), color='red')
    ax[row, 1].set_title(continuous_fields[item])
    ax[row, 1].grid(True)

    ax[row, 2].boxplot(positive[continuous_fields[item]].dropna())
    ax[row, 2].set_title(continuous_fields[item])
    ax[row, 2].grid(True)
    
    row = row + 1

plt.tight_layout()
plt.show()
clinical_test_confidence = {}
for col in continuous_fields:
    confidence_high = stats.norm.interval(0.68, loc=positive[col].dropna().mean(), scale=positive[col].dropna().std())
    confidence_medium = stats.norm.interval(0.95, loc=positive[col].dropna().mean(), scale=positive[col].dropna().std())
    confidence_low = stats.norm.interval(0.99, loc=positive[col].dropna().mean(), scale=positive[col].dropna().std())
    clinical_test_confidence[col + '_high'] = list(confidence_high)
    clinical_test_confidence[col + '_medium'] = list(confidence_medium)
    clinical_test_confidence[col + '_low'] = list(confidence_low)
clinical_test_confidence_df = pd.DataFrame(clinical_test_confidence)
clinical_test_confidence_df
def transform_data(data, column):
    row = ''
    low1, low2 = clinical_test_confidence_df[column + '_low']
    med1, med2 = clinical_test_confidence_df[column + '_medium']
    high1, high2 = clinical_test_confidence_df[column + '_high']
    
    if data and data < low1 and data > low2:
        row = 0
    
    elif data and (data > low1 and data < med1) or (data > med2 and data < low2):
        row = 1
    
    elif data and (data > med1 and data < high1) or (data > high2 and data < med2):
        row = 2
    
    elif data and data > high1 and data < high2:
        row = 3
    
    return row
df_new = pd.DataFrame()
for col in continuous_fields:
    df_new[col] = df[col].apply(lambda row: transform_data(row, col))
fig, ax = plt.subplots(math.ceil(len(categorical_field)/3), 3, figsize=(15,60))
row = 0
for item in range(0, len(categorical_field), 3):
    for col in range(0, 3):
        try:
            positive.fillna('NULL').groupby([categorical_field[item+col]])[categorical_field[item+col]].count().plot.bar(ax=ax[row, col])
            ax[row, col].set_title(categorical_field[item+col])
            ax[row, col].set_xlabel('')
            ax[row, col].grid(True)
        except Exception:
            ax[row, col].axis('off')
    
    row = row + 1

plt.tight_layout()
plt.show()
def simple_score(data):
    score = 1
    for col in data.items():
        if col[1]:
            score = score + col[1]
    
    return score
df['score'] = df_new.apply(lambda row: simple_score(row), axis=1)
fig, ax = plt.subplots(1, 2, figsize=(20,5))
ax[0].hist(df['score'].loc[(df['SARS-Cov-2 exam result'] == 'positive') & (df['score'] != 1)])
ax[0].set_title('Score frequency for positive cases')
ax[0].grid(True)
ax[1].hist(df['score'].loc[(df['SARS-Cov-2 exam result'] == 'negative') & (df['score'] != 1)])
ax[1].set_title('Score frequency for negative cases')
ax[1].grid(True)
plt.show()