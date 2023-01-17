import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r'../input/results/results.csv') 
df.head()
df.shape
df.columns
df.info()
df.describe() 
df.describe(include='object') 
df.nunique()
df['home_team'].unique()
df['home_team'].value_counts()
df['home_team'].value_counts(normalize='True') 