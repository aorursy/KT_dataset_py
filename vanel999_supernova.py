# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import gzip #This module provides a simple interface to compress and decompress files just like the GNU programs gzip and gunzip would.
import pickle #This module implements a fundamental, but powerful algorithm for serializing and de-serializing a Python object structure
import pandas as pd
import numpy as np


def read_data(filename):
    """Read data from pickled file to a pandas dataframe"""
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f)

    X = to_dataframe(data)
    y = pd.get_dummies(X.type == 0, prefix='SNIa', drop_first=True) #Convert categorical variable into dummy/indicator variables
    X = X.drop(columns=['comment', 'type'])

    return X, y


def to_dataframe(data):
    """Converts from a python dictionary to a pandas dataframe"""
    for idx in data:
        sn = data[idx]
        for filt in 'griz':
            sn['mjd_%s' % filt] = np.array(sn[filt]['mjd'])
            sn['fluxcal_%s' % filt] = np.array(sn[filt]['fluxcal'])
            sn['fluxcalerr_%s' % filt] = np.array(sn[filt]['fluxcalerr'])
            del sn[filt]
        sn.update(sn['header'])
        del sn['header']

    return pd.DataFrame.from_dict(data, orient='index')
X, y = read_data('../input/des_train.pkl')
# 5 first rows of the dataframe
X.head()
import matplotlib.pyplot as plt
plt.style.use('seaborn')
%matplotlib inline

DES_FILTERS = 'griz' #Gaussian Filter

def plot_lightcurves(idx):
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
    for id_f, f in enumerate(DES_FILTERS):
        ax = axes[id_f // 2, id_f % 2]
        ax.errorbar(X.iloc[idx]['mjd_%s' % f], 
                    X.iloc[idx]['fluxcal_%s' % f], 
                    X.iloc[idx]['fluxcalerr_%s' % f], 
                    fmt='o')
        ax.set_xlabel('MJD')
        ax.set_ylabel('Calibrated flux')
        ax.set_title('%s-band' % f)
plot_lightcurves(0)
def bazin(time, A, B, t0, tfall, trise):
    X = np.exp(-(time - t0) / tfall) / (1 + np.exp((time - t0) / trise))
    return A * X + B
from scipy.optimize import least_squares

def lightcurve_fit(time, flux):
    scaled_time = time - time.min()
    t0 = scaled_time[flux.argmax()]
    guess = (0, 0, t0, 40, -5)

    errfunc = lambda params: abs(flux - bazin(scaled_time, *params))

    result = least_squares(errfunc, guess, method='lm')

    return result.x
def uncertain_mean(flux, flux_err, w):
    UM = [x / y if y != 0 else x for x, y in zip(flux, flux_err)]
    if w != 0:
        return (1/w) * sum(UM)
    return sum(UM)
def uncertain_moving_average(flux, flux_err, w):
    l = len(flux)
    UMA = []
    if l <= 2*w : 
        UMA = flux
    else:
        UMA = [uncertain_mean(flux[range(i,w+i)], flux_err[range(i,w+i)], w) if i <= l-w else uncertain_mean(flux[range(i,l)], flux_err[range(i,l)], l-i)  for i in range(l)]
    return UMA
def znormalization(UMA):
    array_UMA = np.array(UMA)
    m = np.mean(array_UMA)
    std = np.std(array_UMA)
    zUMA = [(x - m)/std  for x in UMA]
    return zUMA
def plot_lightcurves_with_fit(idx):
    w = 5
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 8))
    for id_f, f in enumerate(DES_FILTERS):
        ax = axes[id_f // 2, id_f % 2]
        
        time = X.iloc[idx]['mjd_%s' % f]
        flux = X.iloc[idx]['fluxcal_%s' % f]
        flux_err = X.iloc[idx]['fluxcalerr_%s' % f]
                
        flux_uma = uncertain_moving_average(flux, flux_err, w)
        
        flux_zuma = znormalization(flux_uma)
        flux_zuma = np.array(flux_zuma)
        
        fit = lightcurve_fit(time, flux_zuma)
        stime = np.arange(time.min(), time.max())
        
        ax.plot(time, flux_zuma, 'o')
        ax.plot(stime, bazin(stime - stime.min(), *fit))
        ax.set_xlabel('MJD')
        ax.set_ylabel('Calibrated flux')
        ax.set_title('%s-band' % f)
plot_lightcurves_with_fit(2)
def preprocessing(data):
    #Window length for computing uncertain moving average filtering
    w = 10
    # Create palceholder for output matrix
    full_params = np.zeros((len(data), 5 * len(DES_FILTERS)))
    # Iterate over supernovae
    for idx, snid in enumerate(data.index):
        params = np.zeros((len(DES_FILTERS), 5))
        # Iterate over filters
        for id_f, f in enumerate(DES_FILTERS):
            time = data.loc[snid, 'mjd_%s' % f]
            flux = data.loc[snid, 'fluxcal_%s' % f]
            flux_err = data.loc[snid, 'fluxcalerr_%s' % f]
            
            flux_uma = uncertain_moving_average(flux, flux_err, w)
            flux_zuma = znormalization(flux_uma)
            flux_zuma = np.array(flux_zuma)
            try:
                params[id_f] = lightcurve_fit(time,  flux_zuma)
            except ValueError:
                # If fit does not converge leave zeros
                continue
        full_params[idx] = params.ravel()
                           
    return full_params
# feature_extractor.py

class FeatureExtractor():
    def __init__(self):
        pass

    def fit(self, X_df, y):
        pass

    def transform(self, X_df):
        return preprocessing(X_df)
preprocessing(X)