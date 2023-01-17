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
va = [7, 6, 5, 4, 3, 3.5, 4.5, 5.5, 6.5]

vb = [5.4, 4.5, 3.5, 2.55, 1.65, 2.1, 3.0, 4.0, 5.0]

current = [350, 330, 320, 300, 280, 280, 310, 320, 340]

tp = [4.9, 6.0, 7.6, 10.5, 16.5, 12.9, 8.9, 6.7, 5.4]
from cmath import pi

import numpy as np

import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import polyfit

import pandas as pd





current = [val/1000 for val in current]

tp = [4.9, 6.0, 7.6, 10.5, 16.5, 12.9, 8.9, 6.7, 5.4]

tp = [i/1000 for i in tp]



a = [1/(2*pi)]



w = [(2*pi)/i for i in tp]



d = {"V_b":vb, "w":w}

table1 = pd.DataFrame(data=d)

table1.index += 1

table1



b, k = polyfit(w, vb, 1)

print("k_b =", k)
plt.plot(np.unique(w), np.poly1d(np.polyfit(w, vb, 1))(np.unique(w)))

plt.show()
i = [320, 400, 445, 500, 400, 520]

tp2 = [6.0, 6.4, 6.5, 6.7, 6.5, 7.1]

# Weirdly enough I didn't have to use to Vb from table 2...


i = [b/1000 for b in i]

tp2 = [a/1000 for a in tp2]

w2 = [(2*pi)/items for items in tp2]

torque = [item*k for item in i]
data2 = {"Torque":torque, "Speed w": w2}

table2 = pd.DataFrame(data = data2)

table2.index = ["released", "FL1", "FL2", "FL3", "EL>RIGHT", "EL>LEFT"]

table2
k1, k2 = polyfit(w2, torque, 1)

plt.plot(np.unique(w2), np.poly1d(np.polyfit(w2, torque, 1))(np.unique(w2)))

plt.show()
print("k_1 =", k1, ", and k_2 =", k2)
#B calculation

b = torque[0]/w2[0]

print("B = ", b)



def Average(lst):

    return sum(lst) / len(lst)



#R_a calcualtion

r_a = []

for index in range(0,8):

    r = (va[index]-vb[index])/current[index]

    r_a.append(r)



print("R_a = ", r_a)

print("R_a average =", Average(r_a))

print("w_no load = ", w2[0], "w_full load =", w2[3])

s_reg = 100*(w2[0] - w2[3])/w2[3]

print("s_regulation = ", s_reg)



print("for chosen S, i(in A not mA)=", i[3], "w = ", w2[3])

p_out = k*i[3]*w[3]

print("P_out =", p_out)



p_in = 6*i[3]

print("P_in =", p_in)



eff = 100*p_out/p_in

print("efficiency =", eff)
k1_theoretical = k*va[0]/r_a[0]

k2_theoretical = k*k/r_a[0]

print("k1_theo =", k1_theoretical)

print("k2_theo =", k2_theoretical)