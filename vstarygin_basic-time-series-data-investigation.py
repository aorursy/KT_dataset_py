# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf

import scipy.stats

from statsmodels.tsa.statespace.sarimax import SARIMAX

import statsmodels.api as sm

import collections





sns.set(style="darkgrid")





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sales_train_validation = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_submission = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")
d_cols = []

rename_cols = {}

for i in range(1,1914):

    d_cols.append(i)

    rename_cols["d_"+str(i)] = i



sales_train_validation = sales_train_validation.rename(columns = rename_cols)
df = sales_train_validation.melt(id_vars = ["id","item_id","dept_id","cat_id","store_id","state_id"],value_vars = d_cols).rename(columns = {"variable":"day","value":"quantity"})
df.head()
plot1 = df[["state_id","day","quantity"]].groupby(["state_id","day"]).sum().reset_index() 

plt.figure(figsize=(18, 6))

ax = sns.lineplot(x = 'day', y = 'quantity', hue="state_id", data=plot1)

del plot1
plot1 = df[["state_id","store_id","day","quantity"]].groupby(["state_id","store_id","day"]).sum().reset_index() 

g = sns.FacetGrid(data = plot1, col="state_id", hue="store_id", height=6, margin_titles=True)

ax = g.map(sns.lineplot, "day", "quantity")

g.add_legend();

del plot1
plot1 = df[["state_id","cat_id","day","quantity"]].groupby(["state_id","cat_id","day"]).sum().reset_index() 

g = sns.FacetGrid(data = plot1, col="state_id", hue="cat_id", height=6, margin_titles=True)

ax = g.map(sns.lineplot, "day", "quantity")

g.add_legend();

del plot1
plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "CA"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

g = sns.FacetGrid(data = plot1, col="store_id", hue="cat_id", height=6, margin_titles=True)

ax = g.map(sns.lineplot, "day", "quantity")

g.add_legend();

del plot1
def Chow_test(X,y,point):

    k=2

    X1 = np.zeros((len(X),2))

    Y1 = np.zeros((len(y),1))

    

    X1[:,0] = 1

    X1[:,1] = X

    Y1[:,0] = y

    

    try:

        weights = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(Y1)

    except:

        return (float("nan"),float("nan"))

        

    Y_reg =  X1.dot(weights)

    Y_reg = np.reshape(Y_reg,(1,-1))[0]

    

    x1 = X[0:point]

    x2 = X[point:len(X)]

    y1 = y[0:point]

    y2 = y[point:len(X)]

    

    X11 = np.zeros((len(x1),2))

    Y11 = np.zeros((len(y1),1))

    X11[:,0] = 1

    X11[:,1] = x1

    Y11[:,0] = y1

    

    try:

        weights1 = np.linalg.inv(X11.T.dot(X11)).dot(X11.T).dot(Y11)

    except:

        return (float("nan"),float("nan"))

    

    y_temp1 = X11.dot(weights1)

    y_temp1 = np.reshape(y_temp1,(1,-1))[0]

    

    

    X21 = np.zeros((len(x2),2))

    Y21 = np.zeros((len(y2),1))

    X21[:,0] = 1

    X21[:,1] = x2

    Y21[:,0] = y2

    

    try:

        weights2 = np.linalg.inv(X21.T.dot(X21)).dot(X21.T).dot(Y21)    

    except:

        return (float("nan"),float("nan"))

    y_temp2 = X21.dot(weights2)

    y_temp2 = np.reshape(y_temp2,(1,-1))[0]

    

    e = sum((y - Y_reg)**2)

    e1 = sum((y1 -y_temp1)**2)

    e2 = sum((y2 -y_temp2)**2)

    

    chow = ((e-e1-e2)/k)/((e1+e2)/(len(y1)+len(y2)-2*k))

    F_stat = scipy.stats.f.ppf(q=1-0.05, dfn=k, dfd=len(y1)+len(y2)-2*k)

    

    return (chow, F_stat)
day_CA_2_FOODS = 0

chow_stat = 0 

plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "CA"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

plot1 = plot1[(plot1["store_id"]=="CA_2")&(plot1["cat_id"]=="FOODS")] 

for i in plot1["day"]:

    (chow, F) = Chow_test(plot1["day"], plot1["quantity"], i) 

    if ((chow > chow_stat) and (chow > F)):

        chow_stat =  chow

        day_CA_2_FOODS = i
print("Trend brake at point",day_CA_2_FOODS,"in store CA_2 in category FOODS.")
df = df.drop(df[(df["store_id"]=="CA_2") & (df["cat_id"]=="FOODS") & (df["day"] < day_CA_2_FOODS)].index)
plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "TX"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

g = sns.FacetGrid(data = plot1, col="store_id", hue="cat_id", height=6, margin_titles=True)

ax = g.map(sns.lineplot, "day", "quantity")

g.add_legend();

del plot1
plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "WI"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

g = sns.FacetGrid(data = plot1, col="store_id", hue="cat_id", height=6, margin_titles=True)

ax = g.map(sns.lineplot, "day", "quantity")

g.add_legend();

del plot1
day_WI_1_FOODS = 0

chow_stat = 0 

plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "WI"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

plot1 = plot1[(plot1["store_id"]=="WI_1")&(plot1["cat_id"]=="FOODS")] 

for i in plot1["day"]:

    (chow, F) = Chow_test(plot1["day"], plot1["quantity"], i) 

    if ((chow > chow_stat) and (chow > F)):

        chow_stat =  chow

        day_WI_1_FOODS = i

        

day_WI_2_FOODS = 0

chow_stat = 0 

plot1 = df[["state_id","store_id","cat_id","day","quantity"]][df["state_id"] == "WI"].groupby(["state_id","store_id","cat_id","day"]).sum().reset_index()

plot1 = plot1[(plot1["store_id"]=="WI_2")&(plot1["cat_id"]=="FOODS")] 

for i in plot1["day"]:

    (chow, F) = Chow_test(plot1["day"], plot1["quantity"], i) 

    if ((chow > chow_stat) and (chow > F)):

        chow_stat =  chow

        day_WI_2_FOODS = i

        

print("Trend brake at point",day_WI_1_FOODS,"in store WI_1 in category FOODS.")

print("Trend brake at point",day_WI_2_FOODS,"in store WI_2 in category FOODS.")
df = df.drop(df[(df["store_id"]=="WI_1") & (df["cat_id"]=="FOODS") & (df["day"] < day_WI_1_FOODS)].index)

df = df.drop(df[(df["store_id"]=="WI_2") & (df["cat_id"]=="FOODS") & (df["day"] < day_WI_2_FOODS)].index)
plot1 = df[df["state_id"] == "CA"][["store_id","cat_id","day","quantity"]].groupby(["store_id","cat_id","day"]).sum().reset_index()

fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "CA_1")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "CA_1 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_1")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "CA_1 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_1")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "CA_1 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "CA_2")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "CA_2 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_2")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "CA_2 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_2")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "CA_2 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "CA_3")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "CA_3 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_3")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "CA_3 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_3")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "CA_3 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "CA_4")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "CA_4 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_4")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "CA_4 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "CA_4")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "CA_4 HOUSEHOLD",ax =axes[2])

del plot1
plot1 = df[df["state_id"] == "TX"][["store_id","cat_id","day","quantity"]].groupby(["store_id","cat_id","day"]).sum().reset_index()

fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "TX_1")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "TX_1 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_1")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "TX_1 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_1")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "TX_1 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "TX_2")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "TX_2 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_2")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "TX_2 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_2")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "TX_2 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "TX_3")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "TX_3 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_3")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "TX_3 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "TX_3")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "TX_3 HOUSEHOLD",ax =axes[2])



del plot1
plot1 = df[df["state_id"] == "WI"][["store_id","cat_id","day","quantity"]].groupby(["store_id","cat_id","day"]).sum().reset_index()

fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "WI_1")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "WI_1 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_1")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "WI_1 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_1")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "WI_1 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "WI_2")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "WI_2 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_2")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "WI_2 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_2")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "WI_2 HOUSEHOLD",ax =axes[2])



fig, axes = plt.subplots(1,3,figsize=(25,5))

fig = plot_acf(plot1[(plot1["store_id"] == "WI_3")&(plot1["cat_id"]=="FOODS")]["quantity"], title = "WI_3 FOODS",ax =axes[0])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_3")&(plot1["cat_id"]=="HOBBIES")]["quantity"], title = "WI_3 HOBBIES",ax =axes[1])

fig = plot_acf(plot1[(plot1["store_id"] == "WI_3")&(plot1["cat_id"]=="HOUSEHOLD")]["quantity"], title = "WI_3 HOUSEHOLD",ax =axes[2])



del plot1
LAG = 7
def  stationarity_check(x,store):

    res1 = sm.tsa.adfuller(x["quantity"].values,regression='c')

    res2 = sm.tsa.adfuller(x["quantity"].values,regression='ct')

    res3 = sm.tsa.adfuller(x["quantity"].values,regression='ctt')

    cat = x["cat_id"].unique()[0]

    if (res1[1] > 0.05):

        print("P-value for", cat, "in store", store, "for model with constant is", res1[1],"Model is not stationary.")

    if (res2[1] > 0.05):

        print("P-value for", cat, "in store", store, "for model with linear trend is", res2[1],"Model is not stationary.")

    if (res3[1] > 0.05):

        print("P-value for", cat, "in store", store, "for model with linear + quadratic is", res3[1],"Model is not stationary.")

    
plot1 = df[df["store_id"] == "CA_1"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"CA_1")

plot1 = df[df["store_id"] == "CA_2"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"CA_2")

plot1 = df[df["store_id"] == "CA_3"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"CA_3")

plot1 = df[df["store_id"] == "CA_4"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"CA_4")



plot1 = df[df["store_id"] == "TX_1"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"TX_1")

plot1 = df[df["store_id"] == "TX_2"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"TX_2")

plot1 = df[df["store_id"] == "TX_3"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"TX_3")





plot1 = df[df["store_id"] == "WI_1"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"WI_1")

plot1 = df[df["store_id"] == "WI_2"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"WI_2")

plot1 = df[df["store_id"] == "WI_3"][["cat_id","day","quantity"]].groupby(["cat_id","day"]).sum().reset_index()

plot1[["cat_id","quantity"]].groupby(["cat_id"]).apply(stationarity_check,"WI_3")

calendar["d"] = [int(i[2:]) for i in calendar["d"]]
calendar = calendar[["wday","month","d","event_name_1","event_type_1","event_name_2","event_type_2","snap_CA","snap_TX","snap_WI"]]
calendar = calendar.rename(columns ={"d":"day"})
def graph(df,calendar,store):

    cat_in_store = df[(df["store_id"] == store)]

    calendar1 = calendar.merge(cat_in_store[["day","cat_id","quantity"]], on = ["day"], how = 'left')

    calendar1 = calendar1.dropna(subset = ["quantity"])

    bar = calendar1[calendar1["event_name_1"].notnull()][["event_name_1","cat_id","quantity"]].groupby(["cat_id","event_name_1"]).sum().reset_index()

    return bar
fig,axes = plt.subplots(2,2,figsize = (24,20))



bar = graph(df,calendar,"CA_1")

axes[0,0].set_title('CA_1')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[0,0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"CA_2")

axes[0,1].set_title('CA_2')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[0,1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"CA_3")

axes[1,0].set_title('CA_3')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[1,0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"CA_4")

axes[1,1].set_title('CA_4')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[1,1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

fig,axes = plt.subplots(1,3,figsize = (25,6))



bar = graph(df,calendar,"TX_1")

axes[0].set_title('TX_1')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"TX_2")

axes[1].set_title('TX_2')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"TX_3")

axes[2].set_title('TX_3')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[2])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



fig,axes = plt.subplots(1,3,figsize = (25,6))



bar = graph(df,calendar,"WI_1")

axes[0].set_title('WI_1')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"WI_2")

axes[1].set_title('WI_2')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = graph(df,calendar,"WI_3")

axes[2].set_title('WI_3')

ax = sns.barplot(x="event_name_1", y="quantity",hue ="cat_id", data=bar,ax = axes[2])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



del bar
def months(df,calendar,store_id):

    cat_in_store = df[df["store_id"] == store_id]

    calendar1 = calendar.merge(cat_in_store[["day","cat_id","quantity"]], on = ["day"], how = 'left')

    calendar1 = calendar1.dropna(subset = ["quantity"])

    g = calendar1[["cat_id","month","quantity"]].groupby(["cat_id","month"]).sum().reset_index()

    return g

fig,axes = plt.subplots(2,2,figsize = (24,20))



bar = months(df,calendar,"CA_1")

axes[0,0].set_title('CA_1')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[0,0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"CA_2")

axes[0,1].set_title('CA_2')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[0,1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"CA_3")

axes[1,0].set_title('CA_3')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[1,0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"CA_4")

axes[1,1].set_title('CA_4')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[1,1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



fig,axes = plt.subplots(1,3,figsize = (25,6))



bar = months(df,calendar,"TX_1")

axes[0].set_title('TX_1')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"TX_2")

axes[1].set_title('TX_2')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"TX_3")

axes[2].set_title('TX_3')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[2])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)





fig,axes = plt.subplots(1,3,figsize = (25,6))



bar = months(df,calendar,"WI_1")

axes[0].set_title('WI_1')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[0])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"WI_2")

axes[1].set_title('WI_2')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[1])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



bar = months(df,calendar,"WI_3")

axes[2].set_title('WI_3')

ax = sns.barplot(x="month", y="quantity",hue ="cat_id", data=bar,ax = axes[2])

ax1 = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)



del bar
def snap(df,calendar,state_id):

    cat_in_store = df[df["state_id"] == state_id]

    calendar1 = calendar.merge(cat_in_store[["day","quantity"]], on = ["day"], how = 'left')

    calendar1 = calendar1.dropna(subset = ["quantity"])

    g = calendar1[["snap_"+state_id,"quantity"]].groupby(["snap_"+state_id]).mean().reset_index()

    return g
bar = snap(df,calendar,"CA")

bar1 = snap(df,calendar,"TX")

bar2 = snap(df,calendar,"WI")
fig,axes = plt.subplots(1,3,figsize = (15,8))

axes[0].set_title('Average snap sales')

axes[1].set_title('Average snap sales')

axes[2].set_title('Average snap sales')

ax = sns.barplot(x="snap_CA", y="quantity", data=bar,ax = axes[0])

ax = sns.barplot(x="snap_TX", y="quantity", data=bar1,ax = axes[1])

ax = sns.barplot(x="snap_WI", y="quantity", data=bar2,ax = axes[2])