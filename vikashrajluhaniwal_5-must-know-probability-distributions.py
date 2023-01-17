import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import plotly.figure_factory as ff

import plotly.graph_objects as go

import plotly.express as px



from scipy.stats import norm

from scipy.stats import probplot 
ais = pd.read_csv("/kaggle/input/australian-athletes-data-set/ais.csv")
hist_data = [ais[ais["sex"]=="m"]["rcc"], ais[ais["sex"]=="f"]["rcc"]]

group_labels = ["male", "female"]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,show_rug=False)

fig['layout'].update(title={"text" : 'Distribution of rcc count based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="rcc",yaxis_title="probability density")

fig.update_layout(width=500,height=500)

fig
hist_data = [ais[ais["sex"]=="m"]["wcc"], ais[ais["sex"]=="f"]["wcc"]]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,show_rug=False)

fig['layout'].update(title={"text" : 'Distribution of wcc count based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="wcc",yaxis_title="probability density")

fig.update_layout(width=500,height=500)
hist_data = [ais[ais["sex"]=="m"]["bmi"], ais[ais["sex"]=="f"]["bmi"]]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,show_rug=False)

fig['layout'].update(title={"text" : 'Distribution of Body mass index(BMI) based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="bmi",yaxis_title="probability density")

fig.update_layout(width=500,height=500)
hist_data = [ais[ais["sex"]=="m"]["pcBfat"], ais[ais["sex"]=="f"]["pcBfat"]]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,show_rug=False)

fig['layout'].update(title={"text" : 'Distribution of percent Body fat(pcBfat) based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="pcBfat",yaxis_title="probability density")

fig.update_layout(width=500,height=500)
hist_data = [ais[ais["sex"]=="m"]["lbm"], ais[ais["sex"]=="f"]["lbm"]]

fig = ff.create_distplot(hist_data, group_labels,show_hist=False,show_rug=False)

fig['layout'].update(title={"text" : 'Distribution of lean body mass(lbm) based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="lbm",yaxis_title="probability density")

fig.update_layout(width=500,height=500)
counts, bin_edges = np.histogram(ais[ais["sex"]=="m"]['rcc'],bins = 100, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig = go.Figure()

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "male"))

counts, bin_edges = np.histogram(ais[ais["sex"]=="f"]['rcc'],bins = 100,density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "female"))

fig['layout'].update(title={"text" : 'CDF of rcc based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="rcc",yaxis_title="probability")

fig.update_layout(width=500,height=500)
counts, bin_edges = np.histogram(ais[ais["sex"]=="m"]['pcBfat'],bins = 200, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig = go.Figure()

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "male"))

counts, bin_edges = np.histogram(ais[ais["sex"]=="f"]['pcBfat'],bins = 200,density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "female"))

fig['layout'].update(title={"text" : 'CDF of pcBfat based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="pcBfat",yaxis_title="probability")

fig.update_layout(width=500,height=500)
counts, bin_edges = np.histogram(ais[ais["sex"]=="m"]['lbm'],bins = 200, density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig = go.Figure()

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "male"))

counts, bin_edges = np.histogram(ais[ais["sex"]=="f"]['lbm'],bins = 200,density = True)

pdf = counts/(sum(counts))

cdf = np.cumsum(pdf)

fig.add_trace(go.Scatter(x=bin_edges[1:], y=cdf,mode='lines',name = "female"))

fig['layout'].update(title={"text" : 'CDF of lbm based on Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="lbm",yaxis_title="probability")

fig.update_layout(width=500,height=500)
fig = ff.create_distplot([ais["ht"]], ["ht"],show_hist=False,show_rug=False)

fig['layout'].update(title={'text':'Distribution of height','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="height",yaxis_title="probability")

fig.update_layout(showlegend = False,width=500,height=500)

fig
fig = ff.create_distplot([ais["wt"]], ["wt"],show_hist=False,show_rug=False)

fig['layout'].update(title={'text':'Distribution of weight','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="weight",yaxis_title="probability")

fig.update_layout(showlegend = False,width=500,height=500)

fig
from scipy.stats import skew

print("Skewness of height parameter : ", skew(ais["ht"]))

print("Skewness of weight parameter : ", skew(ais["wt"]))
fig = ff.create_distplot([ais["ht"]], ["ht"],show_hist=False,show_rug=False,curve_type="normal")

fig['layout'].update(title={'text':'Distribution of height','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="height",yaxis_title="probability")

fig.update_layout(showlegend = False,width=500,height=500)

fig
from scipy.stats import norm

print("% of atheletes heights falling below 165 cm is : ", norm.cdf(165, loc = np.mean(ais["ht"]), scale = np.std(ais["ht"])))

print("% of atheletes heights falling between 165 and 185 cm is : ", norm.cdf(185, loc = np.mean(ais["ht"]), scale = np.std(ais["ht"])) - norm.cdf(165, loc = np.mean(ais["ht"]), scale = np.std(ais["ht"])))

print("% of atheletes heights falling above 185 cm is : ", 1- norm.cdf(185, loc = np.mean(ais["ht"]), scale = np.std(ais["ht"])))
from scipy.stats import probplot

qq = probplot(ais["wt"], dist='norm')

x = np.array([qq[0][0][0],qq[0][0][-1]])

fig = go.Figure()

fig.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1], mode = 'markers',showlegend=False))

fig.add_trace(go.Scatter(x=x,y=qq[1][1] + qq[1][0]*x,showlegend=False,mode='lines'))

fig['layout'].update(title={'text':'Q-Q plot of height','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Theoretical quantiles",yaxis_title="Sample quantiles")

fig.update_layout(width=500,height=500)

fig
from scipy.stats import shapiro

print("p-value obtained from Shapiro-Wilk test : ", shapiro(ais["ht"])[1])
from scipy.stats import kstest

print(kstest(ais["ht"],"norm", args = (np.mean(ais["ht"]), np.std(ais["ht"]))))
print("mean:",np.mean(ais["ht"]))

np.std(ais["ht"])
def Chebyshev(low_limit, mean, std):

    k = (mean - low_limit) / std

    #k = (np.mean(data) - low_limit) / np.std(data) 

    return (1-(1/(k**2)))
print("% of athelete having height between 160.68 and 199.52 cm : ", Chebyshev(160.68,np.mean(ais["ht"]),np.std(ais["ht"])))

print("% of athelete having height between 150.97 and 209.23 cm : ", Chebyshev(150.97,np.mean(ais["ht"]),np.std(ais["ht"])))
fig = ff.create_distplot([ais["ferr"]], ["ferr"],show_hist=False,show_rug=False)

fig['layout'].update(title={'text':'Distribution of height','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="height",yaxis_title="probability")

fig.update_layout(showlegend = False,width=500,height=500)

fig
qq = probplot(ais["ferr"], dist='lognorm',sparams=(1,0))

x = np.array([qq[0][0][0],qq[0][0][-1]])

fig = go.Figure()

fig.add_trace(go.Scatter(x=qq[0][0],y=qq[0][1], mode = 'markers',showlegend=False))

fig.add_trace(go.Scatter(x=x,y=qq[1][1] + qq[1][0]*x,showlegend=False,mode='lines'))

fig['layout'].update(title={'text':'Q-Q plot of height for log-normality','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Theoretical quantiles",yaxis_title="Sample quantiles")

fig.update_layout(width=500,height=500)

fig
from scipy.stats import binom

n= 50 #total number of fixed trials

p= 0.1237 #probability of success

print("probability that exactly two atheletes play basketball : ", binom.pmf(2,50,0.1237))

print("probability that exactly at most 10 atheletes play basketball : ", binom.cdf(10,50,0.1237))

print("probability that exactly at least 20 atheletes play basketball : ", np.round((1- binom.cdf(20,50,0.1237)) + binom.pmf(20,50,0.1237)))
n = 50

p = 0.1237

x1 = np.arange(0, 12,2)

y1 = binom.pmf(x1, n, p)

fig = go.Figure([go.Bar(x=x1, y=y1)])

fig['layout'].update(title={"text" : 'PMF - Binomial Distribution','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="# of successes in 50 trails",yaxis_title="prob.")

fig.update_layout(width=500,height=500)

fig
fig = go.Figure([go.Bar(x=ais["sex"].value_counts().index, y=ais["sex"].value_counts().values)])

fig['layout'].update(title={"text" : 'Distribution of Sex','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="Sex",yaxis_title="count")

fig.update_layout(width=500,height=500)

fig
hist_data = [ais[((ais["wcc"]>=7.70) & (ais["wcc"]<=9.90)) | ((ais["wcc"]>=4.40) & (ais["wcc"]<5.50))]["wcc"]]

group_labels = ["wcc"]

fig = ff.create_distplot(hist_data, group_labels, bin_size=1.10,show_curve=False, show_rug = False,histnorm = "probability")

fig['layout'].update(title={"text" : 'Distribution of wcc','y':0.9,'x':0.5,'xanchor': 'center','yanchor': 'top'}, xaxis_title="wcc",yaxis_title="prob.")

fig.update_layout(width=500,height=500)

fig