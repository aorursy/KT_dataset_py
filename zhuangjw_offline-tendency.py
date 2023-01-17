%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import dask
from dask.diagnostics import ProgressBar

from ipywidgets import interact, IntSlider, Dropdown
ds = xr.open_mfdataset('../input/*.nc4').drop('KPP_RFactive')
ds.dims
ds.nbytes/1e9  # GB
# remove KPP_ prefix
original_name = list(ds.data_vars.keys())
trimed_name = [ s[4:] for s in original_name]
ds.rename(name_dict=dict(zip(original_name, trimed_name)), inplace=True);
# GEOS5 output is upsidedown
ds = ds.sel(lev=slice(None, None, -1))
ds['lev'] = np.arange(ds['lev'].size)
bf_chem_name = [s for s in trimed_name if 'BEFORE_CHEM' in s]
af_chem_name = [s for s in trimed_name if 'AFTER_CHEM' in s]
chem_name = [s[11:] for s in af_chem_name]
ds_bf = ds[bf_chem_name]  # before reation
ds_bf.rename(name_dict=dict(zip(bf_chem_name, chem_name)), inplace=True)

ds_af = ds[af_chem_name]  # after reaction
ds_af.rename(name_dict=dict(zip(af_chem_name, chem_name)), inplace=True);
# tendency by chemistry
ds_chem_tend = ds_af - ds_bf
# tendency by other processes = BEFORE_CHEM(t+1) - AFTER_CHEM(t)
# Note that the actual time step is not 1 hour.

ds_bf_temp = ds_bf.isel(time=slice(1, None)).drop('time') # BEFORE_CHEM(t+1)
ds_af_temp = ds_af.isel(time=slice(0, -1)).drop('time') # AFTER_CHEM(t)

ds_other_tend =  ds_bf_temp - ds_af_temp  # use absolute difference 
ds_other_tend_fct = ds_bf_temp / ds_af_temp  # use scaling factor
# accumulate the tendency
ds_other_tend_cum = ds_other_tend.cumsum(dim='time')
ds_other_tend_fct_cum = ds_other_tend_fct.cumprod(dim='time')
# apply tendency to initial condition
# get a time-series with chemistry tendency removed
# this resembles a ML model that always predicts zero tendency.

# use absolute difference
ds_no_chem_v1 = ds_af.isel(time=0, drop=True) + ds_other_tend_cum

# use scaling factor
ds_no_chem_v2 = ds_af.isel(time=0, drop=True) * ds_other_tend_fct_cum
with ProgressBar():
    no_chem_max_v1, no_chem_min_v1 = dask.compute(ds_no_chem_v1.max(), 
                                                  ds_no_chem_v1.min())
with ProgressBar():
    no_chem_max_v2, no_chem_min_v2 = dask.compute(ds_no_chem_v2.max(), 
                                                  ds_no_chem_v2.min())
with ProgressBar():
    ref_max, ref_min = dask.compute(ds_af.max(), ds_af.min())
def to_series(ds):
    return pd.Series({k: v.item() for k, v in ds.data_vars.items()})
df_range = pd.concat([to_series(ds) for ds in [ref_max, ref_min, 
                                               no_chem_max_v1, no_chem_min_v1,
                                               no_chem_max_v2, no_chem_min_v2]], axis=1)
df_range.columns = ['ref_max', 'ref_min', 'v1_max', 'v1_min', 'v2_max', 'v2_min']

df_range 
# many negative values in v1_min (linear tendency)
# many large values / inf in v2_max (multiplicative tendency)
@interact(t=IntSlider(min=0, max=18, continuous_update=False),
          l=IntSlider(min=0, max=24, continuous_update=False),
          var=Dropdown(options=chem_name))
def plot_layer(var, t, l):
    fig, axes = plt.subplots(1, 3, figsize=[12, 3])
    
    ds_no_chem_v1[var].isel(time=t, lev=l).plot(ax=axes[0])
    axes[0].set_title('linear tendency')
    ds_no_chem_v2[var].isel(time=t, lev=l).plot(ax=axes[1])
    
    axes[1].set_title('multiplicative tendency')
    
    # this not the reference solution without chemistry tendency
    # just to show the typical range of chemicals
    ds_af[var].isel(time=t+1, lev=l).plot(ax=axes[2])
    axes[2].set_title('original data')
plot_layer('NO', 4, 0)  # NO blows up very quickly
plot_layer('NO2', 18, 0)  # same for NO2
plot_layer('O3', 14, 0)  # O3 is OK, except for a few extreme cells in second figure
