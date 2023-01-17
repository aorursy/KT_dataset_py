# data analysis and wrangling

import pandas as pd

from pandas_datareader import data

import numpy as np

import random as rnd

import datetime



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline



# plotly

import plotly

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots

import cufflinks as cf

cf.go_offline()
company_list = pd.read_csv('../input/warren-buffett-us-stock-companies/Company List.csv', sep=';')

sec_form = pd.read_csv('../input/warren-buffett-us-stock-companies/SEC Form 13F.csv', sep=';')

AAPL = pd.read_csv('../input/warren-buffett-us-stock-companies/AAPL.csv')
# Company List

company_list
# SEC Form 13F

sec_form.head()
# Apple Stock

AAPL.head()