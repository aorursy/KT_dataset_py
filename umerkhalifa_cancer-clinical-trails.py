# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Modules for importing data 

import pandas as pd

import numpy as np



# Modules for data visualizaion 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

clinical_trail = pd.read_csv("../input/eligibilityforcancerclinicaltrials/labeledEligibilitySample1000000.csv", sep = "\t")
# Adding label to the data

clinical_trail.columns = ["label", "trail"]



for i in clinical_trail.head():

    print(i)
# Cross checking the dimension of the data 

clinical_trail.shape
# Exploratory data analysis 



# Eligible counts

label_0 = clinical_trail.query("label == '__label__0'").groupby("label")["trail"].count()

# Not Eligible counts

label_1 = clinical_trail.query("label == '__label__1'").groupby("label")["trail"].count()
# Splitting the trail statement into two

clinical_trail[["trail", "diseases"]] = clinical_trail["trail"].str.split(".", n = 1, expand = True)



clinical_trail.head()
# counts of unique cases

clinical_trail["diseases"].value_counts().sort_values(ascending = False)



clinical_trail["trail"].value_counts().sort_values(ascending = False)


# Extracting dominant disease type



clinical_trail["lymphoma"] = clinical_trail["diseases"].str.contains("lymphoma")



clinical_trail["breast_cancer"] = clinical_trail["diseases"].str.contains("breast cancer")
# Extracting and visualizing actions for dominant disease type

# clinical trails on lymphoma cancer

lymphoma = clinical_trail.query("lymphoma == True").groupby(["trail", "label"])["diseases"].count().sort_values(ascending = False).head(10)



plt.figure(figsize = (12,8))

sns.barplot(x = lymphoma.index, y  = lymphoma.values, alpha = 0.7, edgecolor = "b")

plt.ylabel("Action counts")

plt.xticks(rotation = 90)

plt.title("Clinical trails on Lymphoma Cancer")



# clinical trails on breast cancer

breast_cancer = clinical_trail.query("breast_cancer == True").groupby(["trail", "label"])["diseases"].count().sort_values(ascending = False).head(10)



plt.figure(figsize = (12,8))

sns.barplot(x = breast_cancer.index, y  = breast_cancer.values, alpha = 0.7, edgecolor = "b")

plt.ylabel("Action counts")

plt.xticks(rotation = 90)

plt.title("Eligible Clinical trails on Breast Cancer")