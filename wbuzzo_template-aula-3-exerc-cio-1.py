import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Lendo o Dataset
df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
x = df[df['uf'] == 'BA'].nlargest(15, 'total_eleitores')
all_data = [x['gen_feminino'],x['gen_masculino']]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# plot box plot
axes[1].boxplot(all_data)
axes[1].set_title('box plot')

# plot violin plot
axes[0].violinplot(all_data,showmeans=False,showmedians=True)
axes[0].set_title('violin plot')


# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)

plt.violinplot(all_data,showmeans=True, showmedians=False) #default
plt.show()



df = pd.read_csv('../input/BR_eleitorado_2016_municipio.csv', delimiter=',')
df = df[df['uf'] == 'MG'].nlargest(15, 'total_eleitores')
all_data_1 = [df['f_18_20'],df['f_21_24']]
all_data_2 = [df['f_70_79'],df['f_sup_79']]

plt.violinplot(all_data_1,showmeans=True, showmedians=False) #default
plt.title('f_18_20 x f_21_24')
plt.show()

plt.violinplot(all_data_2,showmeans=True, showmedians=False) #default
plt.title('f_70_79 x f_sup_79')
plt.show()
