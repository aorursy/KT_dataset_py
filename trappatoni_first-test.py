# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import sklearn

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



training_test = pd.read_csv("../input/train.csv",iterator = True, chunksize = 10000)
first_chunk = training_test.read(10000)
products = pd.read_csv("../input/producto_tabla.csv")
products.index = products["Producto_ID"]
first_chunk.columns

first_chunk["Producto_ID"].unique()

first_chunk["score"] = first_chunk["Venta_uni_hoy"] - first_chunk["Demanda_uni_equil"]
score.describe()
score.hist()
product_ID = first_chunk.groupby("Producto_ID")

for prod_Id, group in product_ID:
    
    score = group["Venta_uni_hoy"] - group["Demanda_uni_equil"]
    
score = first_chunk["score"]
target = first_chunk["Demanda_uni_equil"]
feature = first_chunk[['Agencia_ID', 'Canal_ID', 'Ruta_SAK', 'Cliente_ID',
       'Producto_ID', 'Venta_uni_hoy', 'Venta_hoy', 'Dev_uni_proxima',
       'Dev_proxima']]
from sklearn import tree
dec_tree = tree.DecisionTreeClassifier()
dec_tree = dec_tree.fit(feature, target)
print(dec_tree.feature_importances_)
print(dec_tree.score(feature,target))
# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(feature, target)

# Print the score of the fitted random forest
print(my_forest.score(feature, target))


