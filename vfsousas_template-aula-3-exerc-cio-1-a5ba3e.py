import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

ds = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv')
dsBA = ds[ds['uf']=='BA'].sort_values(by=['total_eleitores']).head(15)
dsBAFemale = dsBA['gen_feminino']
dsBAMale = dsBA['gen_masculino']
import matplotlib.pyplot as plt
plt.violinplot([dsBAFemale, dsBAMale],showmeans=True, showmedians=True) 
plt.xticks([0,1,2,3], ('', 'Gen Feminino', 'Gen Masculino'))

dsMG = ds[ds['uf']=='MG'].sort_values(by=['total_eleitores']).head(15)
a = dsMG['f_16']+dsMG['f_17']+dsMG['f_18_20']
b = dsMG['f_21_24']
c = dsMG['f_70_79']
d = dsMG['f_sup_79']
plt.xticks([0,1,2,3,4], ('', '16-20', '21-24','70-79', '>79'))
plt.violinplot([a, b, c, d],showmeans=True, showmedians=True) 


ds = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv')
dsBA = ds[ds['uf']=='BA'].sort_values(by=['total_eleitores']).head(15)
dsBAFemale = dsBA['gen_feminino']
dsBAMale = dsBA['gen_masculino']
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].boxplot(dsBAFemale)
axes[1].boxplot(dsBAMale)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
axes[0].boxplot(a)
axes[1].boxplot(b)
axes[2].boxplot(c)
axes[3].boxplot(d)
plt.show()
