%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from ipywidgets import interact, IntSlider, Dropdown

import sklearn
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, Activation

import torch
ds = xr.open_mfdataset('../input/*.nc4').drop('KPP_RFactive')
ds.dims
df = ds.isel(time=[0,1], drop=True).to_dataframe().reset_index(drop=True).rename(columns=lambda s: s.replace('KPP_', ''))

df_y = df.iloc[:, df.columns.str.startswith('AFTER_CHEM')].rename(columns=lambda s: s.replace('AFTER_CHEM_', ''))

df_x = df.iloc[:, df.columns.str.startswith('BEFORE_CHEM')].rename(columns=lambda s: s.replace('BEFORE_CHEM_', ''))

df_in = df.iloc[:, ~df.columns.str.startswith('AFTER_CHEM')].rename(columns=lambda s: s.replace('BEFORE_CHEM_', ''))
varnames = list(df_x.columns)
print(varnames)
%%time
var = 'O3'
#var = 'NO'

fig, axes = plt.subplots(2, 2, figsize=[12, 8])

axes[0,0].scatter(df_x[var]*1e9, df_y[var]*1e9, alpha=0.05)
axes[0,0].set_xlabel('before reaction'); axes[0,0].set_ylabel('after reaction')
axes[0,1].scatter(df_x[var]*1e9, (df_y[var]-df_x[var])*1e9, alpha=0.05)
axes[0,1].set_xlabel('before reaction'); axes[0,1].set_ylabel('tendency')

axes[1,0].hexbin(df_x[var]*1e9, df_y[var]*1e9, bins='log')
axes[1,1].hexbin(df_x[var]*1e9, (df_y[var]-df_x[var])*1e9, bins='log')
fig.suptitle(var, fontsize=20)
@interact(var=Dropdown(options=varnames))
def plot_onevar(var):
    fig, axes = plt.subplots(2, 2, figsize=[12, 8])
    
    axes[0,0].scatter(df_x[var]*1e9, df_y[var]*1e9, alpha=0.05)
    axes[0,0].set_xlabel('before reaction'); axes[0,0].set_ylabel('after reaction')
    axes[0,1].scatter(df_x[var]*1e9, (df_y[var]-df_x[var])*1e9, alpha=0.05)
    axes[0,1].set_xlabel('before reaction'); axes[0,1].set_ylabel('tendency')
    
    axes[1,0].hexbin(df_x[var]*1e9, df_y[var]*1e9, bins='log')
    axes[1,1].hexbin(df_x[var]*1e9, (df_y[var]-df_x[var])*1e9, bins='log')
    fig.suptitle(var, fontsize=20)

@interact(var=Dropdown(options=varnames, label='O3'))
def plot_distribution(var):
    fig, axes = plt.subplots(2, 2, figsize=[12, 8])
    axes[0,0].hist(df_x[var]); axes[0,0].set_title('original; linear')
    axes[0,1].hist(df_y[var]-df_x[var]); axes[0,1].set_title('tendency; linear')
    axes[1,0].hist(df_x[var], log=True); axes[1,0].set_title('original; logbin')
    axes[1,1].hist(df_y[var]-df_x[var], log=True); axes[1,1].set_title('tendency; logbin')
    fig.suptitle(var)
from scipy.stats import kurtosis

for var in ['O3', 'NO', 'NO2', 'CO']:
    print(var, kurtosis(df_x[var]), kurtosis(df_x[var]-df_y[var]), sep='; ')
# a lot better after log transform
plt.hist(np.log(df_x['NO2']))
kurtosis(np.log(df_x['NO2']))
# Try to reduce the kurtosis of O3 tendency (has negative value)

diff = df_x['O3']-df_y['O3']
diff_transform = np.sign(diff)*np.abs(diff)**0.2
plt.hist(diff_transform)
kurtosis(diff_transform)
pvar = 'O3' # pick up a single value to predict for now

idx_pvar = df_in.columns.get_loc(pvar)
idx_pvar
X_all= df_in.values
Y_all = df_y[[pvar]].values 

# r: raw, unscaled
X_train_r, X_test_r, Y_train_r, Y_test_r = train_test_split(X_all, Y_all, test_size=0.2, random_state=42)

# difference/tendency
Y_train_diff_r = Y_train_r - X_train_r[:, idx_pvar:idx_pvar+1]
Y_test_diff_r = Y_test_r - X_test_r[:, idx_pvar:idx_pvar+1]

Y_train_diff_r.shape, Y_test_diff_r.shape
scaler_X = StandardScaler()
scaler_Y_origin = StandardScaler()
scaler_Y_diff = StandardScaler()
X_train = scaler_X.fit_transform(X_train_r)
X_test = scaler_X.transform(X_test_r)

Y_train_origin = scaler_Y_origin.fit_transform(Y_train_r)
Y_test_origin = scaler_Y_origin.transform(Y_test_r)

Y_train_diff = scaler_Y_diff.fit_transform(Y_train_diff_r)
Y_test_diff = scaler_Y_diff.transform(Y_test_diff_r)
# test reverse transform
np.allclose(Y_train_diff_r, scaler_Y_diff.inverse_transform(Y_train_diff))
# Very high kurtotic O3 tendency as before
plt.hist(Y_train_diff);
model_1 = Sequential([
    Dense(100, input_shape=(127,)),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(1),
])

model_1.compile(optimizer='adam',
              loss='mse')

model_1.fit(X_train, Y_train_origin, epochs=10, batch_size=128)

%time Y_train_pred_1 = model_1.predict(X_train)
r2_score(Y_train_pred_1, Y_train_origin)
%time Y_test_pred_1 = model_1.predict(X_test)
r2_score(Y_test_pred_1, Y_test_origin)
plt.scatter(Y_test_pred_1, Y_test_origin, alpha=0.2, s=4)
Y_test_diff_1 = scaler_Y_diff.transform(
    scaler_Y_origin.inverse_transform(Y_test_pred_1) 
    - X_test_r[:, idx_pvar:idx_pvar+1]
)
plt.scatter(Y_test_diff_1, Y_test_diff, alpha=0.2, s=4)
r2_score(Y_test_diff_1, Y_test_diff)
model_2 = Sequential([
    Dense(100, input_shape=(127,)),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(1),
])

model_2.compile(optimizer='adam',
              loss='mse')

model_2.fit(X_train, Y_train_diff, epochs=10, batch_size=128)

%time Y_train_pred_2 = model_2.predict(X_train)
r2_score(Y_train_pred_2, Y_train_diff)
%time Y_test_pred_2 = model_2.predict(X_test)
r2_score(Y_test_pred_2, Y_test_diff)
plt.scatter(Y_test_pred_2, Y_test_diff, alpha=0.2, s=4);
Y_test_original_2 = scaler_Y_origin.transform(
    scaler_Y_diff.inverse_transform(Y_test_pred_2) 
    + X_test_r[:, idx_pvar:idx_pvar+1]
)
plt.scatter(Y_test_original_2, Y_test_origin, alpha=0.2, s=4);
r2_score(Y_test_original_2, Y_test_origin)