# import libraries
from bs4 import BeautifulSoup
from urllib import request
import time
import random
import numpy as np
import pandas as pd
import re
# creat file and write headers
filename = "newegg_desktop_pageN.csv"
f = open(filename, "w", encoding='utf-8')
headers = "Brand,Core,Ram,HDD,SSD,Graphics,Price,Detail \n"
f.write(headers)
# Get HDD informations function
def getHDD(container):
    product_detail = list(container.children)[3].img["alt"]
    p = [r"\d+\s?\w+\s?HDD", r"\d+\w+\s?HDD", r"\d+TB\s?\w?\s?SSHD", r"\d+TB\s?.+\sRPM"]
    for i in p:
        findhdd = re.search(i, product_detail)
        if findhdd != None:
            break
        else:
            continue

    if findhdd != None:
        hdd = findhdd.group()
    else:
        hdd = ""

    return hdd

# Get Core informations function
def getcore(container):
    product_detail = list(container.children)[3].img["alt"]
    findcore = re.search(r"\d.\d+\sGHz", product_detail)
    if findcore != None:
        core = findcore.group()
    else:
        findcore = re.search(r"\d.\d+GHz", product_detail)
        if findcore != None:
            core = findcore.group()
        else:
            core = ""
    return core

# Get Ram informations function
def getram(container):
    product_detail = list(container.children)[3].img["alt"]
    findram = re.search(r"\d+\s?GB\s?DDR\d", product_detail)
    if findram == None:
        ram = ""

    else:
        ram = findram.group()
    return ram

# Get SSD informations function
def getSSD(container):
    product_detail = list(container.children)[3].img['alt']
    p = [r'\d+GB\s\w+\sSSD', 
         r'\d+\s?\w+\s?SSD', 
         r'\d+\w+\s?SSD',
         r'\d+TB\s?\w?\s?SSD',
         r'\d+TB\s?\w+\s?SSD',
         r'x \d+\s?GB',
         r'\d+\sGB\s+\S+\s+SSD',
         r'\d+\sGB\s+\S+\s+\S+\s+SSD',
         r'\d+\sGB\s+\S+\s+\S+\s+\S+\s+SSD',
         r'\d+GB\s+\SSD',
        r'\d+GB\s+\S+\sSSD']
    for i in p:
        findssd = re.search(i,product_detail)
        if findssd != None:
            break
        else:
            continue
            
    if findssd != None:
        ssd = findssd.group()
    else:
        ssd = ''
        
    return ssd
# Get Graphics informations function
def getgraphics(container):
    product_detail = list(container.children)[3].img['alt']
    findgraphics = re.search(r'(NVIDIA|AMD)\s?\w+\s?\w+\s\d+',product_detail)
    
    p = [r'Radeon RX Vega\s\S+',
         r'(NVIDIA|AMD)\s?\w+\s?\w+\s\d+', 
         r'Intel HD Graphics \d+']
    for i in p:
        findgraphics = re.search(i,product_detail)
        if findgraphics != None:
            break
        else:
            continue
    
    if findgraphics != None:
        graphics = findgraphics.group()
    else:
        graphics = ''

    return graphics

# Get price informations function
def getprice(container):
    price_current = container.findAll("li", {"class": "price-current"})
    for p in price_current:
        price_c = p.text
        price_c = re.search(r"\$\d\,?\d+", price_c)
        if price_c == None:
            price = ""
        else:
            price = price_c.group()
            price = price.replace(",", "")
        return price

# Scraping 100 pages from newegg
"""
for i in range(1, 100):
    my_url = "https://www.newegg.ca/Desktop-Computers/SubCategory/ID-10/Page-{}?Tid=6737&PageSize=96&order=BESTMATCH".format(
        i
    )
    uClient = request.urlopen(my_url)
    page_html = uClient.read()
    uClient.close()
    page_soup = BeautifulSoup(page_html, "html.parser")
    containers = page_soup.findAll("div", {"class": "item-container "})
    for container in containers:
        product_detail = list(container.children)[3].img["alt"].replace(",", "  ")
        brand = product_detail.split()[0]
        core = getcore(container)
        ram = getram(container)
        HDD = getHDD(container)
        SSD = getSSD(container)
        graphics = getgraphics(container)
        price = getprice(container)
        f.write(
            brand
            + ","
            + core
            + ","
            + ram
            + ","
            + HDD
            + ","
            + SSD
            + ","
            + graphics
            + ","
            + price
            + ","
            + product_detail
            + "\n"
        )
        time.sleep(0.5 + random.random())
    time.sleep(10 + random.random())
    print(i)
"""
f.close()
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# load data from csv file
dataset = pd.read_csv("../input/newegg_desktop_pageN - V3.csv", encoding="latin-1", dtype=str)

dataset.info()

dataset.head(6)

# clean label Price and transform into float
dataset.Price = dataset.Price.str.replace(",", "")
dataset.Price = dataset.Price.str.replace("$", "")
dataset.Price = dataset.Price.astype(float)

# drop na
print("length of dataset before dropna of column Price:", len(dataset))
dataset.dropna(thresh=1, subset=["Price"], inplace=True)
print("length of dataset after dropna of column Price:", len(dataset))

# add one column as sequence for agg use
dataset["seq"] = [i for i in range(len(dataset.Brand))]
dataset[:5]

# clean Feature: Brand
dataset.Brand = dataset.Brand.str.lower()
dataset.Brand = dataset.Brand.str.replace("amsdell", "dell")
dataset.Brand = dataset.Brand.str.replace("certified", "refurbished")
dataset.Brand = dataset.Brand.str.strip()
brand = dataset.groupby(by="Brand", as_index=False).agg({"seq": pd.Series.nunique})
brand.seq.sum()
brand[brand.seq >= 30]

# rename brand as 'others' for all brands which sequence less than 30 —— expect alienware and apple of course
other_brand_list = brand[brand.seq < 30].Brand
for i in other_brand_list:
    if i != "alienware" and i != "apple":
        dataset.Brand = dataset.Brand.str.replace(i, "others")

other_brand = ["otherstel",
    "othersfothersite",
    "others40t",
    "others40",
    "viewsonothers",
    "wothersdows",
    "othersdustrial",
    "thotherskcentre",
    "versuspower",
    "othersest",
    "optiplex",
    "others7",
    "othersplex",
    "hp-elitedesk-800-g3-i5-7500t-2-7ghz-256gb-ssd-4gb-mothersi-others-hp-warranty"]

for b in other_brand:
    dataset.Brand = dataset.Brand.str.replace(b, "others")

dataset.Brand = dataset.Brand.str.replace("cyberpowerothers", "cyberpower")
dataset.Brand = dataset.Brand.str.replace("cybertronothers", "cybertron")
dataset.Brand = dataset.Brand.str.replace("vothers", "others")
dataset.Brand.unique()

brand2 = dataset.groupby(by="Brand", as_index=False).agg({"seq": pd.Series.nunique})
# t = brand2[brand2.seq >= 5]
check = dataset.groupby("Brand", as_index=False)["Price"].mean()
v = check[check.Brand.isin([i for i in brand2.Brand])]
v = v.sort_values(by="Price")
v

v["value"] = v.Price / np.mean(v.Price)
v.drop(columns="Price")

# plot Brand Average Price
v.plot(
    x='Brand',
    y='Price',
    kind="bar",
    figsize=(12,6),
    use_index=True,
    title='Brand Average Price',
    grid=True,
    legend=True,
    fontsize=12,
    colormap=None,
)

# find all i/Ryzen series in dataset
reg = [r"Ryzen \d+", r" i\d"]
irlist = []
for i in dataset.Detail:
    for j in reg:
        find = re.search(j, i)
        if find == None:
            f = ""
        else:
            f = find.group()
        irlist.append(f)
set(irlist)

# Creat new column Iseries to record i/Ryzen series number
dataset["Iseries"] = "0"
iseries = [" i3", " i5", " i7", " i9", "Ryzen 3", "Ryzen 5", "Ryzen 7"]
for i in iseries:
    index2 = dataset.Iseries[dataset.Detail.str.contains(i)].index
    dataset.loc[index2, "Iseries"] = i[-1]
dataset.Iseries.unique()

dataset.Iseries = dataset.Iseries.astype(float)

# check how many examples contains i/Ryzen series record
iseries = [" i3", " i5", " i7", " i9", "Ryzen 3", "Ryzen 5", "Ryzen 7"]
countI = dataset.Detail[dataset.Detail.str.contains("|".join(iseries))]

countIn = dataset.Detail[dataset.Detail.str.contains("|".join(iseries)) == False]
print(
    "lengh of examples contains i/Ryzen series record: ",
    len(countIn),
    "lengh of examples not contains i/Ryzen series record: ",
    len(countI),
    "lengh of dataset: ",
    len(dataset),
)

# dataset[(dataset.Detail.str.contains('|'.join(iseries)) == False) & (dataset.Price <= 5000) & (dataset.Price >= 500)]
dataset.columns
# Creat column Workstation to mark if the example is Workstation: 1 for Workstation and 0 for not
dataset["Workstation"] = "0"
indexw = dataset.Workstation[dataset.Detail.str.contains("Workstation")].index
dataset.loc[indexw, "Workstation"] = "1"
dataset.Workstation = dataset.Workstation.astype(float)

# Clean data for feature Core
dataset.Core = dataset.Core.str.replace("GHz", "")
dataset.Core = dataset.Core.str.strip()
dataset.Core = dataset.Core.str.replace("0 ", "")
dataset.Core = dataset.Core.str.replace("2-7", "2.7")
dataset.Core = dataset.Core.str.replace("5 3", "3")
dataset.Core = dataset.Core.str.strip()
dataset.Core.unique()

# drop nan
print("length of dataset before dropna of column Core:", len(dataset))
dataset.dropna(thresh=1, subset=["Core"], inplace=True)
print("length of dataset after dropna of column Core:", len(dataset))

dataset.Core = dataset.Core.astype(float)

# creat new feature DDR
dataset["DDR"] = dataset.Ram.str[-1]
dataset.DDR.unique()

dataset.Ram.unique()

# clean Ram
clean_data = ["GB DDR2", "GB DDR3", "GB DDR4", "GB DDR5"]
for cd in clean_data:
    dataset.Ram = dataset.Ram.str.replace(cd, "")
dataset.Ram = dataset.Ram.str.strip()
dataset.Ram.unique()

# drop nan
print("length of dataset before dropna of column Ram:", len(dataset))
dataset.dropna(thresh=1, subset=["Ram"], inplace=True)
print("length of dataset after dropna of column Ram:", len(dataset))

dataset.Ram = dataset.Ram.astype(float)
dataset.DDR = dataset.DDR.astype(float)

# clean Graphics, first check the unique Graphics values
dataset.Graphics.unique()

len(dataset[dataset.Graphics.notna()])

score = pd.read_csv("../input/videocardbenchmark.csv", encoding="latin-1", dtype=str)
score.head()

# Update Graphics format
dataset.Graphics = dataset.Graphics.str.strip("NVIDIA")
dataset.Graphics = dataset.Graphics.str.strip("AMD")
dataset.Graphics = dataset.Graphics.str.lower()
dataset.Graphics = dataset.Graphics.str.strip()

# Merge Score of Graphics Model into dataset
dataset = dataset.merge(score, left_on="Graphics", right_on="model", how="left")
dataset.columns

dataset.score = dataset.score.fillna("0")
dataset.score = dataset.score.astype(float)

dataset = dataset.drop(columns="Graphics")

dataset = dataset.drop(columns="model")


dataset.SSD.fillna(value="0", inplace=True)
len(dataset)

dataset.HDD.unique()

# clean HDD
clean_data = ["HDD", "SSHD", "GB", "RPM", "7200", "SSD"]

for cd in clean_data:
    dataset.HDD = dataset.HDD.str.replace(cd, "")

dataset.HDD = dataset.HDD.str.strip()
dataset.HDD.unique()

# keep cleaning...
# dataset.drop(dataset[dataset.HDD == ''].index,inplace=True)
dataset.HDD = dataset.HDD.str.replace("4 1TB", "1TB")
dataset.HDD = dataset.HDD.str.replace("3 1TB", "1TB")
dataset.HDD = dataset.HDD.str.replace("4 2TB", "2TB")
dataset.HDD = dataset.HDD.str.replace("2 80", "80")
dataset.HDD = dataset.HDD.str.replace("4 2TB", "2TB")
dataset.HDD = dataset.HDD.str.replace("2 1TB", "1TB")
dataset.HDD = dataset.HDD.str.replace("4 3TB", "3TB")
dataset.HDD = dataset.HDD.str.replace("4 4TB", "4TB")
dataset.HDD = dataset.HDD.str.replace("G", "")
dataset.HDD = dataset.HDD.str.replace(" ", "")
dataset.HDD = dataset.HDD.str.replace("TB", "000")
dataset.HDD.unique()

dataset.drop(dataset[dataset.HDD == ""].index, inplace=True)
dataset.HDD = dataset.HDD.fillna("0")
dataset.HDD.unique()

# drop nan
print("length of dataset before dropna of column HDD:", len(dataset))
dataset.dropna(thresh=1, subset=["HDD"], inplace=True)
print("length of dataset after dropna of column HDD:", len(dataset))

dataset.HDD = dataset.HDD.astype(float)

# clean SSD
dataset.SSD.unique()

dataset[dataset.SSD == "2 PCIe SSD"].values
dataset.SSD = dataset.SSD.str.replace("2 PCIe SSD", "256")
dataset[
    (dataset.SSD == "2 SATA SSD")
    & dataset.Detail.str.contains("DDR4 32 GB M.2 SATA SSD")
].values
dataset.SSD = dataset.SSD.str.replace("2 SATA SSD", "32")

# clean SSD
clean_data = [
    " GB SSD",
    "GB SSD",
    "SSD",
    "G",
    "PCIe",
    "BSATA",
    "BNVMe",
    "NVMe",
    "SATA",
    "4250",
    "4240",
    "x",
    "B",
    "3128",
    "3 ",
    "4 ",
    "nboard",
    "O",
]

for cd in clean_data:
    dataset.SSD = dataset.SSD.str.replace(cd, "")

dataset.SSD = dataset.SSD.str.strip()

dataset.SSD = dataset.SSD.str.replace("T", "000")
dataset.SSD = dataset.SSD.str.replace(" ", "")
dataset.SSD = dataset.SSD.str.replace("M.2", "")
# dataset.SSD = dataset.SSD.str.replace('TB','000')
# dataset.SSD = dataset.SSD.str.strip()
dataset.SSD.unique()

# dataset = dataset.sort_values(by = 'pricing',ascending=True, inplace=True)
dataset[:10]

# set type as float
dataset.SSD = dataset.SSD.astype(float)

len(dataset)

dataset = dataset.drop(columns="Detail")
dataset = dataset.drop(columns="seq")

dataset[:5]

# OneHotEncoder: using pd.get_dummies
dataset = pd.get_dummies(dataset)
dataset[:5]
dataset = dataset.sort_values(by="Price")
dataset.drop(dataset[dataset.Price <= 800].index, inplace=True)
dataset.drop(dataset[dataset.Price >= 5000].index, inplace=True)
len(dataset)

dataset.columns

from pandas.plotting import scatter_matrix

scatter_matrix = scatter_matrix(
    dataset[
        ["Core", "Iseries", "Workstation", "Ram", "DDR", "HDD", "SSD", "score", "Price"]
    ],
    alpha=0.2,
    figsize=(17, 17),
    diagonal="kde",
)

# X = zip(dataset.Core,dataset.HDD,dataset.Ram,dataset.SSD)
X = dataset.score
Y = dataset.Price
plt.plot(Y, "bo", alpha=0.5, markersize=2)

plt.plot(X, Y, "mo", alpha=0.5, markersize=2)
plt.legend()

# Plot heatmap
import seaborn as sns

cols = ["Core", "Iseries", "Workstation", "Ram", "DDR", "HDD", "SSD", "score", "Price"]
corrcoef_map = np.corrcoef(dataset[cols].values.T)
fig, ax = plt.subplots(figsize=(8, 8))
hm = sns.heatmap(
    corrcoef_map,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=cols,
    xticklabels=cols,
    ax=ax,
)

dataset.columns

import seaborn as sns

cols = [
    "Price",
    "Brand_acer",
    "Brand_alienware",
    "Brand_apple",
    "Brand_asus",
    "Brand_cyberpower",
    "Brand_cybertron",
    "Brand_dell",
    "Brand_hp",
    "Brand_lenovo",
    "Brand_msi",
    "Brand_others",
    "Brand_refurbished",
]
cm = np.corrcoef(dataset[cols].values.T)
fig, ax = plt.subplots(figsize=(18, 18))
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=cols,
    xticklabels=cols,
    ax=ax,
)

dataset = dataset.drop(
    columns=[
        "Workstation",
        "Brand_acer",
        "Brand_alienware",
        "Brand_apple",
        "Brand_cyberpower",
        "Brand_dell",
        "Brand_msi",
        "Brand_refurbished",
    ]
)

dataset.head()

# Standard processing
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit_transform(dataset)

# Split Training/Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset.loc[:, dataset.columns != "Price"], dataset.loc[:, "Price"], test_size=0.3
)

# resort y_test, and using the same order sorting x
order = y_test.argsort(axis=0)
y_test = y_test.values[order]
X_test = X_test.values[order, :]

# Creat Predict and Plot function for ML methods
from sklearn import metrics


def try_different_method(method):
    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)

    maer = 1 - np.mean(abs(y_pred - y_test) / y_test)
    mse = metrics.mean_squared_error(y_pred, y_test)
    r2 = metrics.r2_score(y_pred, y_test)

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(y_pred)),
        y_test,
        "ro-",
        markersize=4,
        label="listing price",
        alpha=0.8,
    )
    plt.plot(
        np.arange(len(y_pred)),
        y_pred,
        "bo-",
        markersize=4,
        label="predict price",
        alpha=0.5,
    )

    plt.grid()
    plt.title("MSE: %f" % mse)
    print("mean_squared_error: %f" % mse)
    print("r2: %f" % r2)
    print("mean_abs_error_rate: %f" % maer)
    plt.legend()
    return mse

result = []

# KNN - Choose hyper parameters
from sklearn import neighbors

# knn = neighbors.KNeighborsRegressor()
mse_list = []
for i in range(1, 6):
    knn = neighbors.KNeighborsRegressor(
        n_neighbors=i,
        weights="uniform",
        algorithm="auto",
        leaf_size=100,
        p=1,
        metric="minkowski",
        metric_params=None,
        n_jobs=1,
    )

    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mse = metrics.mean_squared_error(y_pred, y_test)
    mse_list.append([i, mse])
    print("mean_squared_error for n_neighbors %f: %f" % (i, mse))

pick = pd.DataFrame(mse_list, columns=["nneighbors", "MSE"])
m = pick[pick.MSE == pick.MSE.min()]
n = m.nneighbors.values[0]
m

# KNN
from sklearn import neighbors

# knn = neighbors.KNeighborsRegressor()
knn = neighbors.KNeighborsRegressor(
    n_neighbors=n,
    weights="uniform",
    algorithm="auto",
    leaf_size=100,
    p=1,
    metric="minkowski",
    metric_params=None,
    n_jobs=1,
)

mse_score = try_different_method(knn)
# result.append(m.MSE.values[0])
result.append(("KNN", mse_score))

# DecisionTreeRegressor - Choose hyper parameters
from sklearn import tree

for i in range(10, 100):
    tree_reg = tree.DecisionTreeRegressor(
        criterion="mse",
        splitter="best",
        max_depth=i,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        random_state=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        presort=False,
    )
    tree_reg.fit(X_train, y_train)
    y_pred = tree_reg.predict(X_test)
    mse = metrics.mean_squared_error(y_pred, y_test)
    mse_list.append([i, mse])

pick = pd.DataFrame(mse_list, columns=["maxdepth", "MSE"])
m = pick[pick.MSE == pick.MSE.min()]
n = m.maxdepth.values[0]
m

# DecisionTreeRegressor
from sklearn import tree

tree_reg = tree.DecisionTreeRegressor(
    criterion="mse",
    splitter="best",
    max_depth=n,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    presort=False,
)
mse_score = try_different_method(tree_reg)
# result.append(m.MSE.values[0])
result.append(("Decision Tree", mse_score))

# RandomForestRegressor
from sklearn import ensemble

rf = ensemble.RandomForestRegressor(
    n_estimators=50,
    criterion="mse",
    max_depth=None,
    max_features="auto"
)
mse_score = try_different_method(rf)
result.append(("Random Forest", mse_score))

# GBRT
# gbrt = ensemble.GradientBoostingRegressor(n_estimators=100)
gbrt = ensemble.GradientBoostingRegressor(
    loss="ls",
    learning_rate=0.9,
    n_estimators=200,
    subsample=1.0,
    criterion="friedman_mse",
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
    max_depth=3,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    init=None,
    random_state=None,
    max_features=None,
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    presort="auto",
)
mse_score = try_different_method(gbrt)
result.append(("gbrt", mse_score))

# LinearRegression
from sklearn import linear_model

linear_reg = linear_model.LinearRegression(
    fit_intercept=False, normalize=False, copy_X=True, n_jobs=1
)
mse_score = try_different_method(linear_reg)
result.append(("Linear Regression", mse_score))

# svr = svm.SVR(C=10, gamma=1, kernel="linear")
# try_different_method(svr)

# svr = svm.SVR(C=10, gamma=1, kernel="poly")
# try_different_method(svr)

# SVM - Use GridSearchCV choose hyper parameters
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV

grid_params = {
    "kernel": ["rbf"],
    "gamma": [10 ** i for i in range(-2, 1)],
    "C": [3 ** i for i in range(4, 10)],
}
clf = GridSearchCV(SVR(), grid_params, cv=5, scoring="neg_mean_squared_error")
clf.fit(X_train, y_train)
b = clf.best_params_
print("Best parameter values: %r\n" % clf.best_params_)

means = clf.cv_results_["mean_test_score"]
stds = clf.cv_results_["std_test_score"]
params = stds, clf.cv_results_["params"]

# SVM
from sklearn import svm

svr = svm.SVR(C=b["C"], gamma=b["gamma"], kernel=b["kernel"])
mse_score = try_different_method(svr)
result.append(("SVM", mse_score))

clf.best_params_

comparison = pd.DataFrame(result, columns=["method", "MSE"])
comparison

comparison.plot(
    x="method", kind="bar", grid=True, title="MSE comparison between each method"
)

dataset.head()

dataset = dataset.drop(columns=["Iseries", "score"])
result2 = []

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit_transform(dataset)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset.loc[:, dataset.columns != "Price"], dataset.loc[:, "Price"], test_size=0.3
)

# resort y_test, and using the same order sorting x
order = y_test.argsort(axis=0)
y_test = y_test.values[order]
X_test = X_test.values[order, :]

mse_score = try_different_method(knn)
# result.append(m.MSE.values[0])
result2.append(("KNN", mse_score))

mse_score = try_different_method(tree_reg)
# result.append(m.MSE.values[0])
result2.append(("Decision Tree", mse_score))

mse_score = try_different_method(rf)
result2.append(("Random Forest", mse_score))

mse_score = try_different_method(gbrt)
result2.append(("gbrt", mse_score))

mse_score = try_different_method(linear_reg)
result2.append(("Linear Regression", mse_score))

mse_score = try_different_method(svr)
result2.append(("SVM", mse_score))

comparison2 = pd.DataFrame(result2, columns=["method", "MSE"])
comparison2

comparison
m = pd.merge(
    left=comparison2, right=comparison, on="method", suffixes=("_iteration1", "_iteration2")
)

m.plot(
    x="method",
    y=["MSE_iteration1", "MSE_iteration2"],
    kind="bar",
    grid=True,
    title="MSE comparison between each method"
)

