import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df = df.where(df['uf']=='BA').dropna().nlargest(15,'total_eleitores')
plt.violinplot([df['total_eleitores'],df['gen_feminino'],df['gen_masculino']])
plt.xticks([1.0,1.5,2.0,2.5,3.0],['Total','','Fem','','Masc'])
plt.show()

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

df = df.where(df['uf']=='MG').dropna().nlargest(15,'total_eleitores')
plt.violinplot([df['f_18_20'],df['f_21_24'],df['f_70_79'],df['f_sup_79']])
plt.xticks([1.0,1.5,2.0,2.5,3.0,3.5,4.0],['18-20','','21-24','','70-79','','+ 79'])
plt.show()
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df = df.where(df['uf']=='BA').dropna().nlargest(15,'total_eleitores')
plt.boxplot([df['gen_feminino'],df['gen_masculino']])
plt.xticks([1.0,1.5,2.0],['Fem','','Masc'])
plt.show()


df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')

df = df.where(df['uf']=='MG').dropna().nlargest(15,'total_eleitores')
plt.boxplot([df['f_18_20'],df['f_21_24'],df['f_70_79'],df['f_sup_79']])
plt.xticks([1.0,1.5,2.0,2.5,3.0,3.5,4.0],['18-20','','21-24','','70-79','','+ 79'])
plt.show()