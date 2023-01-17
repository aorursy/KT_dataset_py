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
flist = ["../input/charging_one.txt","../input/charging_two.txt","../input/charging_three.txt","../input/discharging_one.txt","../input/discharging_two.txt","../input/discharging_three.txt"]
wlist = ["charging1.csv","charging2.csv","charging3.csv","discharging1.csv","discharging2.csv","discharging3.csv"]
def tocsv(fname, f2name):
    #fname = "../input/charging_one.txt"
    line2 = []
    with open(fname) as f:
        line = f.readlines()
    for l in line:
        line2.append(l.replace("\t",","))
    line2.pop(0)

    #f2name = "charging1.csv"
    csv = open(f2name, "w") 
    for l in line2:
        csv.write(l)
        
for i in range(0,6):
    tocsv(flist[i],wlist[i])
    print(wlist[i])
