# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
#from  stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
#from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
#from nltk.corpus import stopwords
ps = PorterStemmer()
#stop_words = set(stopwords.words('english'))
#stemmer = SnowballStemmer('english')
import matplotlib.pyplot as plt
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

# loading data
training_data = pd.read_csv("../input/train.csv", encoding="ISO-8859-1")
testing_data = pd.read_csv("../input/test.csv", encoding="ISO-8859-1")
attribute_data = pd.read_csv("../input/attributes.csv")
descriptions = pd.read_csv("../input/product_descriptions.csv")


# concatenating and merging product and brands name to test and training dataset to preform data cleansing analysis
all_train_test = pd.concat((training_data, testing_data), axis=0, ignore_index=True)
num_train_obs = testing_data.shape[0] #keeping track of the number of obs in the training set
all_train_test_desc = pd.merge(all_train_test, descriptions, how='left', on='product_uid')

attribute_data.name.value_counts()

brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
all_train_test_desc_band = pd.merge(all_train_test_desc, brand_names, how='left', on='product_uid')


brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
all_train_test_desc_band = pd.merge(all_train_test_desc, brand_names, how='left', on='product_uid')

#Product measurements
Product_Width = attribute_data[attribute_data.name == "Product Width (in.)"][["product_uid", "value"]].rename(columns={"value": "Product_Width"})
Product_Height = attribute_data[attribute_data.name == "Product Height (in.)"][["product_uid", "value"]].rename(columns={"value": "Product_Height"})
Product_Depth = attribute_data[attribute_data.name == "Product Depth (in.)"][["product_uid", "value"]].rename(columns={"value": "Product_Depth"})
Product_Weight = attribute_data[attribute_data.name == "Product Weight (lb.)"][["product_uid", "value"]].rename(columns={"value": "Product_Weight"})
w_h = pd.merge(Product_Height, Product_Width, how='outer', on='product_uid')
w_h_d = pd.merge(w_h, Product_Depth, how='outer', on='product_uid')
w_h_d_We= pd.merge(w_h_d, Product_Weight, how='outer', on='product_uid')
all_train_test_desc_band_P = pd.merge(all_train_test_desc_band, w_h_d_We, how='left', on='product_uid')
#Assembled measurements
Assembled_Height = attribute_data[attribute_data.name == "Assembled Height (in.)"][["product_uid", "value"]].rename(columns={"value": "Assembled_Height"})
Assembled_Width = attribute_data[attribute_data.name == "Assembled Width (in.)"][["product_uid", "value"]].rename(columns={"value": "Assembled_Width"})
Assembled_Depth = attribute_data[attribute_data.name == "Assembled Depth (in.)"][["product_uid", "value"]].rename(columns={"value": "Assembled_Depth"})
Comm_Res = attribute_data[(attribute_data.name == "Commercial / Residential") | (attribute_data.name == "Commercial/Residential Use") | (attribute_data.name == "Commercial") | (attribute_data.name == "Residential/Commercial/industrial Use")][["product_uid", "value"]].rename(columns={"value": "Comm_Res"})
ENERGY_STAR = attribute_data[(attribute_data.name == "ENERGY STAR Certified")|(attribute_data.name=="Energy Star Qualified") ][["product_uid", "value"]].rename(columns={"value": "ENERGY_STAR"})
aw_ah = pd.merge(Assembled_Height, Assembled_Width, how='outer', on='product_uid')
aw_ah_ad = pd.merge(aw_ah, Assembled_Depth, how='outer', on='product_uid')
ass_com_res = pd.merge(aw_ah_ad, Comm_Res, how='outer', on='product_uid')
ass_com_res_estar = pd.merge(ass_com_res, ENERGY_STAR, how='outer', on='product_uid')

#Color
color= attribute_data[(attribute_data.name=="Finish Family") |(attribute_data.name=="Finish") |(attribute_data.name=="Color/Finish") |(attribute_data.name=="Color Family") | (attribute_data.name=="Color")| (attribute_data.name=="Fixture Color/Finish")| (attribute_data.name=="Finish Family")| (attribute_data.name=="Color/Finish Family")][["product_uid", "value"]].rename(columns={"value": "Colour"})
Voltage= attribute_data[(attribute_data.name=="Voltage (volts)")|(attribute_data.name=="Wattage (watts)")][["product_uid", "value"]].rename(columns={"value": "Voltage"})

all_train_test_desc_band_P_A = pd.merge(all_train_test_desc_band_P, ass_com_res_estar, how='left', on='product_uid')

pat=r'Color|Finish'
colour=attribute_data.name.str.extract(pat, expand=True)
colour.value_counts()