import pandas as pd

pd.set_option('display.max_columns', 50)



from matplotlib import pyplot

# import matplotlib.pylab as plt

import matplotlib.pyplot as plt



from datetime import datetime,timedelta



import numpy as np

from numpy import *



import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
def _from_ordinal(x, tz=None):

    ix = int(x)

    dt = datetime.fromordinal(ix)

    remainder = float(x) - ix

    hour, remainder = divmod(24 * remainder, 1)

    minute, remainder = divmod(60 * remainder, 1)

    second, remainder = divmod(60 * remainder, 1)

    microsecond = int(1e6 * remainder)

    if microsecond < 10:

        microsecond = 0  # compensate for rounding errors

    dt = datetime(dt.year, dt.month, dt.day, int(hour), int(minute),

                  int(second), microsecond)

    if tz is not None:

        dt = dt.astimezone(tz)



    if microsecond > 999990:  # compensate for rounding errors

        dt += timedelta(microseconds=1e6 - microsecond)



    return dt
# READING DATA - Patient 1

pat1_values_original = pd.read_csv('/home/varun/Desktop/dm/assignment1/DataFolder/CGMSeriesLunchPat1.csv')



pat1_ts_original = pd.read_csv('/home/varun/Desktop/dm/assignment1/DataFolder/CGMDatenumLunchPat1.csv')
# REPLACING MISSING DATA WITH N/A AND CONVERTING TIMESERIES FORMAT

pat1_ts_original.fillna(1,inplace=True)

# pat1_values_df.fillna(0,inplace=True)
# pat1_values_df.head()
# CONVERTING TIMESERIES FORMAT

pat1_ts_parsed = pat1_ts_original.applymap(lambda ts : _from_ordinal(ts))
# pat1_ts_parsed.head()
# ROWS & COLUMNS(Original)

pat1_values_original.shape

pat1_ts_parsed.shape
# TS - 

# 1. DECREASES TO THE RIGHT IN A SINGLE DAY

# 2. DECREASES TO THE BOTTOM IN A SINGLE MONTH(ish)

pat1_ts_updated = pat1_ts_parsed.iloc[::-1]

pat1_ts_updated = pat1_ts_updated.iloc[:, ::-1]

pat1_ts_updated.head()
# CGM VALUES - 

# 1. DECREASES TO THE RIGHT IN A SINGLE DAY

# 2. DECREASES TO THE BOTTOM IN A SINGLE MONTH(ish)

pat1_values_updated = pat1_values_original.iloc[::-1]

pat1_values_updated = pat1_values_updated.iloc[:, ::-1]

pat1_values_updated.head()
# ROWS & COLUMNS(Updated)

pat1_values_updated.shape

pat1_ts_updated.shape
# PATIENT 1, DAY 1

plt.xlim( max(pat1_ts_updated.iloc[:1,:].values.flatten())- timedelta(minutes=150), max(pat1_ts_updated.iloc[:1,:].values.flatten()))

plt.plot(pat1_ts_updated.iloc[:1,:].values.flatten(),pat1_values_updated.iloc[:1,:].values.flatten())
# PATIENT 1, DAY 2

plt.xlim( max(pat1_ts_updated.iloc[1:2,:].values.flatten())- timedelta(minutes=150), max(pat1_ts_updated.iloc[1:2,:].values.flatten()))

plt.plot(pat1_ts_updated.iloc[1:2,:].values.flatten(),pat1_values_updated.iloc[1:2,:].values.flatten())
corr = pat1_values_updated.corr()

pat1_values_updated.corr().shape
# GENERATE A MASK FOR THE UPPER TRIANGLE

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# SET UP THE MATPLOTLIB FIGURE

f, ax = plt.subplots(figsize=(11, 9))



# GENERATE A CUSTOM DIVERGING COLORMAP

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# DRAW THE HEATMAP WITH THE MASK & CORRECT ASPECT RATIO

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# CREATING DF COPY

pat1_values_df_features = pat1_values_df.copy()
pat1_values_df_features['max'] = pat1_values_df.max(axis = 1, skipna=True)

pat1_values_df_features['min'] = pat1_values_df.min(axis = 1, skipna=True)

pat1_values_df_features['max-min'] = pat1_values_df_features['max'] - pat1_values_df_features['min']
pat1_values_df_features['mean'] = pat1_values_df.mean(axis = 1, skipna=True)
pat1_values_df_features['std'] = pat1_values_df.std(axis = 1, skipna=True)
pat1_values_df_features['skew'] = pat1_values_df.skew(axis = 1, skipna=True)
# OBSERVATIONS

pat1_values_df_features.head()
# ADDING TARGET - 'meal', since all labels are positive

pat1_values_df_features['target'] = 'meal'

pat1_values_df_features.head()
features = ['max', 'min', 'max-min', 'mean', 'std', 'skew']

# SEPARATING FEATURES

features_unstd = pat1_values_df_features.loc[:, features].values

# SEPARATING TARGET

target = pat1_values_df_features.loc[:,['target']].values

# STANDARDIZING FEATURES

features_std = StandardScaler().fit_transform(features_unstd)

print(features_std)
# APPLYING PCA with n=None

pca = PCA(n_components=None)

principal_components = pca.fit(features_std)



print(principal_components.explained_variance_ratio_)
# APPLYING PCA with n=2

pca = PCA(n_components=2)

principal_components_2 = pca.fit(features_std)



print(principal_components_2.explained_variance_ratio_)
# TODO: convert array to DF

principal_df.shape
# CONCATENATING PRINCIPLE COMPONENTS WITH TARGET

df_final = pd.concat([principal_df, pat1_values_df_features[['target']]], axis = 1)

df_final.head()
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = ['meal', 'not-meal']

colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):

    indicesToKeep = df_final['target'] == target

    ax.scatter(df_final.loc[indicesToKeep, 'principal component 1']

               , df_final.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
# CORRELATION MATRICES

corr_matrix_std = np.corrcoef(features_std.T)

print('Correlation matrix using standardized data\n\n', corr_matrix_std)



corr_matrix_unstd = np.corrcoef(features_unstd.T)

print('\n\nCorrelation matrix using base unstandardized data\n\n', corr_matrix_unstd)



# COVARIANCE MATRIX(Standardized data)

mean_vec = np.mean(features_std, axis=0)

cov_matrix = (features_std - mean_vec).T.dot((features_std - mean_vec)) / (features_std.shape[0]-1)

print('\n\nCovariance matrix \n\n', cov_matrix)