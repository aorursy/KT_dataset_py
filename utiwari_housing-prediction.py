#Before running this code make sure that you have downloaded the data to your kaggle directory

#The data used here is downloaded from the following link which can be simply pasted in the link section of add data button in kaggle:

#"https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.tgz"

import os

import tarfile

tgz_path = "../input/housing.tgz"

housing_path="/kaggle/working/housing"

os.makedirs(housing_path)

housing_tgz = tarfile.open(tgz_path)

housing_tgz.extractall(path=housing_path)

housing_tgz.close()

import os

os.listdir("/kaggle/working/housing")

import pandas as pd

def load_housing_data(housing_path="/kaggle/working/housing"):

    csv_path = os.path.join(housing_path, "housing.csv")

    return pd.read_csv(csv_path)

housing=load_housing_data()
housing.latitude.max()

housing.head()

housing.describe()