# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import requests

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
plt.close("all")

# Raw data
url_c="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
sc=requests.get(url_c).content
confirmed=pd.read_csv(io.StringIO(sc.decode('utf-8')))
url_d="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
sd=requests.get(url_d).content
death=pd.read_csv(io.StringIO(sd.decode('utf-8')))
url_r="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
sr=requests.get(url_r).content
recovered=pd.read_csv(io.StringIO(sr.decode('utf-8')))

# France confirmed
mask_fr=confirmed["Country/Region"]=="France"
mask_2=pd.isna(confirmed["Province/State"])
confirmed_fr=confirmed[mask_fr & mask_2]

# France deaths
mask_fr=death["Country/Region"]=="France"
mask_2=pd.isna(death["Province/State"])
death_fr=death[mask_fr & mask_2]

# France recovered
mask_fr=recovered["Country/Region"]=="France"
mask_2=pd.isna(recovered["Province/State"])
recovered_fr=recovered[mask_fr & mask_2]

# start date
start='3/24/20'
start_idx=confirmed_fr.T.index.tolist().index(start)

# extract daily world
def extract_daily(df, world_pop, colname):
    mask_df=df["Country/Region"].isin(world_pop["Country"].values) | df["Country/Region"].isin(["US", "Korea, South"])
    mask_2=pd.isna(df["Province/State"])
    df_world=df[mask_df & mask_2]
    df_world=df_world.iloc[:,[1, -1]]
    total_df = df[mask_2].iloc[:,-1].sum()
    append_df = pd.DataFrame([['World', total_df]], columns=df_world.columns)
    df_world = df_world.append(append_df, ignore_index=True)
    df_world.columns = ["Country", colname]
    return df_world

# plot function
def plot_with_lin_trend(df, xcol, ycol, std_dev=False):
    z = np.polyfit(x=df.loc[:,xcol], y=df.loc[:,ycol], deg=1)
    p = np.poly1d(z)
    print(z)
    df['trendline'] = p(df.loc[:,xcol])
    df[ycol].plot(kind='line', legend=True), df['trendline'].plot(legend=True)
    if std_dev:
        df['sigma'] = df[ycol].std()
        df['sigma'].plot(legend=True)
    return z

# plot function using index
def plot_with_lin_trend_idx(df, ycol):
    z = np.polyfit(x=df.index, y=df.loc[:,ycol], deg=1)
    p = np.poly1d(z)
    print(z)
    df['trendline'] = p(df.index)
    df[ycol].plot(legend=True), df['trendline'].plot(legend=True)
    return z
data = {
    'C': confirmed_fr.T[126][start:],
    'D': death_fr.T[126][start:],
    'R': recovered_fr.T[113][start:]
}
data_frame = pd.DataFrame(data=data)
plt.figure()
data_frame.plot(logy=False, kind='line')
print(data_frame.iloc[(data_frame.size%3-5):])
mask = data_frame["C"] != 0 & data_frame["D"].any()
clean_df = data_frame[mask]
mortality = 100*clean_df["D"]/clean_df["C"]
epr = 100*(clean_df["D"]+clean_df["R"])/clean_df["C"]
mortality_df = pd.DataFrame({'M':mortality.astype('float'), 'd': pd.Series(data=range(0, mortality.size), index=mortality.index, dtype='float')})
epr_df = pd.DataFrame({'M':epr.astype('float'), 'd': pd.Series(data=range(0, epr.size), index=epr.index, dtype='float')})
m_trend = plot_with_lin_trend(mortality_df, 'd', 'M')
e_trend = plot_with_lin_trend(epr_df, 'd', 'M')
proj = (90 - e_trend[1])/e_trend[0] - epr_df["d"][-1]
print(epr_df.size)
print("Actual days before progression converges : {0} days".format(proj))
vc_s=(clean_df["C"][clean_df.iloc[1].name:].values-clean_df["C"][clean_df.iloc[0].name:clean_df.iloc[-2].name].values)
vd_s=(clean_df["D"][clean_df.iloc[1].name:].values-clean_df["D"][clean_df.iloc[0].name:clean_df.iloc[-2].name].values)
vr_s=(clean_df["R"][clean_df.iloc[1].name:].values-clean_df["R"][clean_df.iloc[0].name:clean_df.iloc[-2].name].values)

vc_s_norm = vc_s / np.mean(vc_s)
vd_s_norm = vd_s / np.mean(vd_s)
vr_s_norm = vr_s / np.mean(vr_s)

speed_df = pd.DataFrame({'VC':vc_s_norm.astype('float'), 'VD':vd_s_norm.astype('float'), 'VR':vr_s_norm.astype('float'), 'd': pd.Series(data=range(0, vc_s_norm.size), index=clean_df["C"][clean_df.iloc[1].name:].index, dtype='float')})
vc_trend = plot_with_lin_trend(speed_df, 'd', 'VC', True)
vd_trend = plot_with_lin_trend(speed_df, 'd', 'VD', True)
vr_trend = plot_with_lin_trend(speed_df, 'd', 'VR', True)
ac_s=(speed_df["VC"][speed_df.iloc[1].name:].values-speed_df["VC"][speed_df.iloc[0].name:speed_df.iloc[-2].name].values)
ad_s=(speed_df["VD"][speed_df.iloc[1].name:].values-speed_df["VD"][speed_df.iloc[0].name:speed_df.iloc[-2].name].values)
ar_s=(speed_df["VR"][speed_df.iloc[1].name:].values-speed_df["VR"][speed_df.iloc[0].name:speed_df.iloc[-2].name].values)

ac_s_norm = ac_s
ad_s_norm = ad_s
ar_s_norm = ar_s

acc_df = pd.DataFrame({'AC':ac_s_norm.astype('float'), 'AD':ad_s_norm.astype('float'), 'AR':ar_s_norm.astype('float'), 'd': pd.Series(data=range(0, ac_s_norm.size), index=speed_df["VC"][speed_df.iloc[1].name:].index, dtype='float')})
ac_trend = plot_with_lin_trend(acc_df, 'd', 'AC', std_dev=True)
ad_trend = plot_with_lin_trend(acc_df, 'd', 'AD', True)
ar_trend = plot_with_lin_trend(acc_df, 'd', 'AR', True)
world_pop=pd.read_csv("../input/world-population/world_pop.csv")
c_world=extract_daily(confirmed, world_pop, 'C')
d_world=extract_daily(death, world_pop, 'D')
r_world=extract_daily(recovered, world_pop, 'R')
n_world=world_pop.sort_values(by=['Country'])[world_pop['Country'] != "China"].reset_index(drop=True)
n_world.columns = ['Country', 'N']
c100k = 100000*c_world['C']/n_world['N']
d100k = 100000*d_world['D']/n_world['N']
prog = 100*(d_world['D']+r_world['R'])/c_world['C']
world_df=pd.DataFrame({
    'Country': c_world["Country"],
    'N': n_world['N'],
    'C': c_world['C'],
    'D': d_world['D'],
    'R': r_world['R'],
    'C100K': c100k,
    'D100K': d100k,
    'D/C': 100*d_world['D']/c_world['C'],
    '(D+R)/C': prog,
    'score': np.log10(c100k*d100k*100/prog)
}, index=c_world.index)
world_df.sort_values(by=['D100K'], ascending=False)
world_df.to_csv(r'world_score.csv', index = False)
