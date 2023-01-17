## Importa umas bibliotecas que vão ser usadas.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

import statsmodels.tsa.stattools as st
df = pd.read_excel('../input/DadosR.xlsx')

# Aqui mostra as primeiras linhas do dataframe

df.head()
# Realiza o teste de granger. Alterando o Maxlag, você pode deslocar as séries temporalmente e comparará-las como se houvesse um lag entre elas.

# Veja documentação aqui: http://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.grangercausalitytests.html

# Obs: os espaços no nome da coluna PBR são porque você deixou assim no Excel.

st.grangercausalitytests(df[['BOND', 'PBR  ']], maxlag=1)
st.grangercausalitytests(df[['BOND', 'PBR  ']], maxlag=2)
## gera gráfico das duas séries

%matplotlib inline

from matplotlib.pyplot import figure

figure(num=None, figsize=(12, 6), dpi=80)



fig = plt.figure()

ax = fig.add_subplot(111)

ax.plot(df['DATA'], df['BOND'], 'r')

ax2 = ax.twinx()

ax2.plot(df['DATA'], df['PBR  '], 'g')

ax.set_ylabel("BOND")

ax2.set_ylabel("PBR")

ax.legend(loc=0)

ax2.legend(loc=0)

ax.grid()

plt.show()


