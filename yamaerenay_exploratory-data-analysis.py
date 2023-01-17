import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
!pip install ppscore
import ppscore as pps
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
with open("/kaggle/input/leagues.json", "r") as read_file:
    data = json.load(read_file)
def random_cmap():
    cmaps = ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r']
    return cmaps[int(np.random.random(1) * len(cmaps))]
df11 = pd.DataFrame(data[20]["daily_results"]["2020-07-27"])
df12 = pd.DataFrame(data[20]["daily_results"]["2020-08-01"])
df21 = pd.DataFrame(data[21]["daily_results"]["2020-07-27"])
df22 = pd.DataFrame(data[21]["daily_results"]["2020-08-01"])
df_1 = pd.concat([df11.set_index("Team")["Pts"], df12.set_index("Team")["Pts"]], axis = 1)
df_2 = pd.concat([df21.set_index("Team")["Pts"], df22.set_index("Team")["Pts"]], axis = 1)
df_1.columns = df_2.columns = ["2020-07-27", "2020-08-01"]

fig = plt.figure(figsize = (18, 10))
ax = fig.add_subplot(2, 2, 1)
ax_2 = fig.add_subplot(2, 2, 2)
ax_3 = fig.add_subplot(2, 2, 3)
ax_4 = fig.add_subplot(2, 2, 4)

ax.set_title("Points, Serie A")
ax.set_xlabel("Team")
ax.set_ylabel("Points")
df_1.plot(kind = "bar", ax = ax, cmap = "coolwarm") 

ax_2.set_title("Points, Serie B")
ax_2.set_xlabel("Team")
ax_2.set_ylabel("Points")
df_2.plot(kind = "bar", ax = ax_2, cmap = "brg")

ax_3.set_xlabel("Points")
df_1.plot(kind = "kde", ax = ax_3, cmap = "coolwarm") 

ax_4.set_xlabel("Points")
df_2.plot(kind = "kde", ax = ax_4, cmap = "brg")
def form(df_1, df_2):
    df = pd.DataFrame(list(df_2["Form"].values), columns = ['sixth', 'fifth', 'fourth', 'third', 'second', 'first'])
    df["seventh"] = [x[::6][0] for x in df_1["Form"].values]
    df = df[['seventh', 'sixth', 'fifth', 'fourth', 'third', 'second', 'first'][::-1]]
    df["Team"] = df_1["Team"].values
    df.set_index("Team", inplace = True)
    xs = []
    for y in range(len(df.index)):
        ys = []
        for x in range(len(df.columns)):
            nums = df.iloc[y, x].split("-")
            ys.append(int(nums[0]) - int(nums[1]))
        xs.append(ys)
    df = pd.DataFrame(data = xs, columns = df.columns, index = df.index)
    for col in df.columns:
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    return df
fig = plt.figure(figsize = (18, 10))
ax = fig.add_subplot(1, 2, 1)
ax_2 = fig.add_subplot(1, 2, 2)
ax.set_title("Form of teams in Italy Serie A")
ax.set_xlabel("Team")
ax.set_ylabel("Form")
form(df11, df12).plot(kind = "bar", ax = ax, stacked = True, cmap = "Wistia")

ax_2.set_title("Form of teams in Italy Serie B")
ax_2.set_xlabel("Team")
ax_2.set_ylabel("Form")
form(df21, df22).plot(kind = "bar", ax = ax_2, stacked = True, cmap = "winter")
def predictivePlotScoreMatrix(df):
    df = df.drop("Form", 1)
    corr = pps.matrix(df)
    corr_df = pd.DataFrame(corr["ppscore"].values.reshape(len(df.columns), len(df.columns)), index = df.columns.values, columns = df.columns.values)
    fig = plt.figure(figsize = (18, 10))
    ax = fig.subplots()
    ax.set_title("Correlation Matrix")
    sns.heatmap(corr_df, cmap = "Greens", annot = True, ax = ax)
    plt.show()
predictivePlotScoreMatrix(df11)
def compareFeatures(df_1, df_2, cmaps, *features):
    fig = plt.figure(figsize = (18, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)
    ax.set_title("Serie A, COMPARISON: " + " - ".join(features))
    df_1.set_index("Team")[list(features)].plot(cmap = cmaps[0], stacked = True, kind = "bar", ax = ax)
    ax_2.set_title("Serie B, COMPARISON: " + " - ".join(features))
    df_2.set_index("Team")[list(features)].plot(cmap = cmaps[1], stacked = True, kind = "bar", ax = ax_2)
compareFeatures(df11, df21, ["inferno", "magma"],"W", "GF", "Pts")
def winDrawLose(df_1, df_2, cmaps):
    features = ["W", "D", "L"]
    df_1 = df_1.copy()
    df_2 = df_2.copy()
    for feature in features:
        df_1[feature + "_"] = df_1[feature]/df_1[features].sum(axis = 1)
        df_2[feature + "_"] = df_2[feature]/df_2[features].sum(axis = 1)
    fig = plt.figure(figsize = (18, 10))
    ax = fig.add_subplot(1, 2, 1)
    ax_2 = fig.add_subplot(1, 2, 2)
    ax.set_title("Serie A, COMPARISON: " + " - ".join(features))
    df_1.set_index("Team")[[x + "_" for x in features]].plot(cmap = cmaps[0], stacked = True, kind = "bar", ax = ax)
    ax_2.set_title("Serie B, COMPARISON: " + " - ".join(features))
    df_2.set_index("Team")[[x + "_" for x in features]].plot(cmap = cmaps[1], stacked = True, kind = "bar", ax = ax_2)
winDrawLose(df11, df21, ["rainbow", "cividis"])