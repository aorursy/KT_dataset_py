# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# 2011-2017

dfx = pd.read_csv("../input/cleanx.csv")

dfx = dfx[3:10].reset_index()



X_Fed = dfx[["CT_Federal", "CTC_Federal", "CTL_Federal", "ST_Federal", "STC_Federal", "STL_Federal"]]

X_ON = dfx[["CT_ON", "CTC_ON", "CTL_ON", "ST_ON", "STC_ON", "STL_ON"]]

X_BC = dfx[["CT_BC", "CTC_BC", "CTL_BC", "ST_BC", "STC_BC", "STL_BC"]]

X_AB = dfx[["CT_AB", "CTC_AB", "CTL_AB", "ST_AB", "STC_AB", "STL_AB"]]

X_QC = dfx[["CT_QC", "CTC_QC", "CTL_QC", "ST_QC", "STC_QC", "STL_QC"]]

X_MB = dfx[["CT_MB", "CTC_MB", "CTL_MB", "ST_MB", "STC_MB", "STL_MB"]]

X_Fed
dfy1 = pd.read_csv("../input/cleany1.csv")

dfy1 = dfy1[3:].reset_index()



# y1

y1_Fed = dfy1[["SE_Federal", "TE_Federal"]]

y1_ON = dfy1[["SE_ON", "TE_ON"]]

y1_BC = dfy1[["SE_BC", "TE_BC"]]

y1_AB = dfy1[["SE_AB", "TE_AB"]]

y1_QC = dfy1[["SE_QC", "TE_QC"]]

y1_MB = dfy1[["SE_MB", "TE_MB"]]



# y2

dfy2 = pd.read_csv("../input/cleany2.csv")

dfy2 = dfy2[1:].reset_index()

y2_Fed = dfy2[["SB_Federal", "TB_Federal"]]

y2_ON = dfy2[["SB_ON", "TB_ON"]]

y2_BC = dfy2[["SB_BC", "TB_BC"]]

y2_AB = dfy2[["SB_AB", "TB_AB"]]

y2_QC = dfy2[["SB_QC", "TB_QC"]]

y2_MB = dfy2[["SB_MB", "TB_MB"]]
import statsmodels.api as sm



X = sm.add_constant(X_Fed)



model = sm.OLS(y1_Fed["SE_Federal"]+y2_Fed["SB_Federal"], X).fit()



model.summary()
import statsmodels.api as sm



X = sm.add_constant(X_ON)



model = sm.OLS(y1_ON["SE_ON"]+y2_ON["SB_ON"], X).fit()



model.summary()
import statsmodels.api as sm



X = sm.add_constant(X_BC)



model = sm.OLS(y1_BC["SE_BC"]+y2_BC["SB_BC"], X).fit()



model.summary()
import statsmodels.api as sm



X = sm.add_constant(X_AB)



model = sm.OLS(y1_AB["SE_AB"]+y2_AB["SB_AB"], X).fit()



model.summary()
import statsmodels.api as sm



X = sm.add_constant(X_QC)



model = sm.OLS(y1_QC["SE_QC"]+y2_QC["SB_QC"], X).fit()



model.summary()
import statsmodels.api as sm



X = sm.add_constant(X_MB)



model = sm.OLS(y1_MB["SE_MB"]+y2_MB["SB_MB"], X).fit()



model.summary()