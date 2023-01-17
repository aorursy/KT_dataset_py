import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%pylab inline

pylab.rcParams['figure.figsize'] = (10, 6)



limit_rows   = 7000000

df           = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,

                                                    "ind_nuevo":str,

                                                    "ult_fec_cli_1t":str,

                                                    "indext":str}, nrows=limit_rows)
unique_ids   = pd.Series(df["ncodpers"].unique())

limit_people = 1.2e4

unique_id    = unique_ids.sample(n=limit_people)

df           = df[df.ncodpers.isin(unique_id)]

df.describe()