# Basic library

import numpy as np

import pandas as pd

import gc

import os



# visualization

from matplotlib import pyplot as plt

plt.style.use("fivethirtyeight")

import seaborn as sns



# Factor analyze

!pip install factor_analyzer

from factor_analyzer import FactorAnalyzer



# PCA

from sklearn.decomposition import PCA



# Coluster analyze

from sklearn.cluster import KMeans



# file

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
path = "/kaggle/input/electric-motor-temperature"



data = pd.read_csv(os.path.join(path, "pmsm_temperature_data.csv"))
data.head()
# dtype change

float_col = ['ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque', 'i_d', 'i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']

int_col = ["profile_id"]



for f in float_col:

    data[f] = data[f].astype("float32")

for i in int_col:

    data[i] = data[i].astype("int16")
data.info()
# profile id num

id_num = data["profile_id"].unique()



# create time columns, 2Hz

list_time = []

for n in id_num:

    df_samp = data[data["profile_id"]==n]

    # create time columns

    time_ = list(np.arange(0,len(df_samp))*0.5)

    list_time = list_time + time_



data["time"] = list_time
# garvage collect

gc.collect()
# Create columns



# Motor work load

data["Motor_work"] = (data["motor_speed"] - data["motor_speed"].min()) * (data["torque"] - data["torque"].min())



# electric work, voltage * current

data["w_d"] = data["u_d"]*data["i_d"]

data["w_q"] = data["u_q"]*data["i_q"]
# Columns

temp_col = ["ambient", "coolant", "pm", "stator_yoke", "stator_tooth", "stator_winding"]

elec_col = ["u_d", "u_q", "i_d", "i_q", "w_d", "w_q"]

work_col = ["motor_speed", "torque", "Motor_work"]

all_col = temp_col + elec_col + work_col
# define heat map, feature correlation

def corr_matrix(data, colname):

    cm = np.corrcoef(data[colname].T)

    plt.figure(figsize=(10,10))

    hm = sns.heatmap(cm, 

                     cbar=True,

                     annot=True,

                     square=True,

                     cmap="bwr",

                     fmt='.2f',

                     annot_kws={"size":10},

                     yticklabels=colname,

                     xticklabels=colname,

                     vmax=1,

                     vmin=-1)

    

# define pair plot

def pair_plot(data, colname, sample_num=1000):

    samp = data[colname].sample(sample_num)

    sns.pairplot(samp)
# plot

pair_plot(data=data, colname=temp_col, sample_num=1000)
# correlation matrix

corr_matrix(data, temp_col)
# plot

pair_plot(data=data, colname=elec_col, sample_num=1000)
# correlation matrix

corr_matrix(data, elec_col)
# plot

pair_plot(data=data, colname=work_col, sample_num=1000)
# correlation matrix

corr_matrix(data, work_col)
# plot

pair_plot(data=data, colname=all_col, sample_num=1000)
# correlation matrix

corr_matrix(data, all_col)
# garvage collect

gc.collect()
# define time series, comparing 2 params

def time_series_2_comp(data, colname1, colname2):

    fig, ax = plt.subplots(13,4, figsize=(5*4,5*13))

    plt.subplots_adjust(hspace=0.3)

    for i in range(0,13):

        for j in range(0,4):

            id_ = id_num[i*4+j]

            samp = data[data["profile_id"]==id_]

            ax[i,j].plot(samp["time"], samp[colname1], label=colname1)

            ax[i,j].plot(samp["time"], samp[colname2], label=colname2)

            ax[i,j].set_title("profile_id : " + str(id_))

            ax[i,j].legend()
### pm and motor work

time_series_2_comp(data, "pm", "Motor_work")
# scree plot

# calculate eigen values

eigen_vals = sorted(np.linalg.eigvals(data[all_col].corr()), reverse=True)



# plot

plt.figure(figsize=(10,6))

plt.plot(eigen_vals, 's-')

plt.xlabel("factor")

plt.ylabel("eigenvalue")
# features num = 4

# Create instance

fa = FactorAnalyzer(n_factors=4, rotation="varimax", impute="drop")



# Fitting

fa.fit(data[all_col])

result = fa.loadings_

colnames = all_col



# Visualization by heatmap

plt.figure(figsize=(10,10))

hm = sns.heatmap(result, cbar=True, annot=True, cmap='bwr', fmt=".2f", 

                 annot_kws={"size":10}, yticklabels=colnames, xticklabels=["factor1", "factor2", "factor3", "factor4"], vmax=1, vmin=-1, center=0)

plt.xlabel("factors")

plt.ylabel("variables")
# Create instance

pca = PCA(n_components=4)



# Fitting

pca_result = pca.fit_transform(data[all_col])

pca_result = pd.DataFrame(pca_result, columns=("pca1", "pca2", "pca3", "pca4"))

pca_result.head()
# visualization by plot

x = pca_result["pca1"]

y = pca_result["pca2"]

c = data["profile_id"]



plt.figure(figsize=(10,8))

plt.scatter(x, y, c=c, alpha=0.8, s=1)

plt.xlabel("pca1")

plt.ylabel("pca2")

plt.colorbar()
# garvage collect

gc.collect()
# elbow

distortions = []

for i in range(1,11):

    km = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=100, random_state=10)

    km.fit(data[all_col])

    distortions.append(km.inertia_)

    

# Plotting distortions

plt.figure(figsize=(10,6))

plt.plot(range(1,11), distortions, marker='o')

plt.xlabel("Number of clusters")

plt.xticks(range(1,11))

plt.ylabel("Distortion")
# Create instance

kmeans = KMeans(n_clusters=4, max_iter=30, init="k-means++", random_state=10)



# Fitting

kmeans.fit(data[all_col])



# output

cluster = kmeans.labels_



# visualization by plot

x = pca_result["pca1"]

y = pca_result["pca2"]

c = cluster



plt.figure(figsize=(10,8))

plt.scatter(x, y, c=c, alpha=0.8, s=1)

plt.xlabel("pca1")

plt.ylabel("pca2")

plt.colorbar()