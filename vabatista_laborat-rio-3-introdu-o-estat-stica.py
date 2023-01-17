%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd



from scipy import stats

import numpy as np  

import scipy as sp





df_train = pd.read_csv('../input/lab3_train_no_nulls_no_outliers.csv')

df_train.head(2)
df_train.describe()
tarifa_homens = df_train[df_train['Sex']=='male']['Fare'].values

tarifa_mulheres = df_train[df_train['Sex']=='female']['Fare'].values
print('qtd. homens: ', tarifa_homens.shape[0])

print('qtd. mulheres: ', tarifa_mulheres.shape[0])



print('média de tarifa homens: ', np.mean(tarifa_homens))

print('média de tarifa mulhers: ', np.mean(tarifa_mulheres))



##Visualizando as distribuições

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))



ax0.hist(tarifa_homens)

ax0.set_title('Tarifas Homens')



ax1.hist(tarifa_mulheres)

ax1.set_title('Tarifas Mulheres')
plt.boxplot([tarifa_homens, tarifa_mulheres], showmeans=True)

plt.show()
print("p-value da hipótese das tarifas dos homens serem normalmente distribuidos ", stats.normaltest(tarifa_homens)[1])

print("p-value da hipótese das tarifas das mulheres serem normalmente distribuidos ", stats.normaltest(tarifa_mulheres)[1])
def intervalo_conf(dados, significancia=0.95):

    a = 1.0*np.array(dados)

    n = len(a)

    m, se = np.mean(a), stats.sem(a)

    h = se * stats.t.ppf((1+significancia)/2., n-1)

    return m-h, m+h
print("Intervalo de confiança com 95% da tarifa dos homens = ", intervalo_conf(tarifa_homens))

print("Intervalo de confiança com 95% da tarifa das mulheres = ", intervalo_conf(tarifa_mulheres))
z_stat, p_val = stats.mannwhitneyu(tarifa_homens, tarifa_mulheres)  

print("O p-value do teste Mann-Whitney-Wilcoxon entre os Dados = ", p_val)
idade_homens = df_train[df_train['Sex']=='male']['Age'].values

idade_mulheres = df_train[df_train['Sex']=='female']['Age'].values
print('média de idade homens: ', np.mean(idade_homens))

print('média de idade mulhers: ', np.mean(idade_mulheres))



##Visualizando as distribuições

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(10, 4))



ax0.hist(idade_homens)

ax0.set_title('Idades Homens')



ax1.hist(idade_mulheres)

ax1.set_title('Idades Mulheres')
print("p-value da hipótese das idades dos homens serem normalmente distribuidos ", stats.normaltest(idade_homens)[1])

print("p-value da hipótese das idades das mulheres serem normalmente distribuidos ", stats.normaltest(idade_mulheres)[1])
def intervalo_conf_z(dados, significancia=0.05):

    a = 1.0*np.array(dados)

    n = len(a)

    m, sd = np.mean(a), np.std(a)

    h = -stats.norm.ppf(q=significancia) * (sd/((n-1)**0.5))

    return m-h, m+h
print("Intervalo de confiança com 95% da tarifa dos homens = ", intervalo_conf(idade_homens))

print("Intervalo de confiança com 95% da tarifa das mulheres = ", intervalo_conf_z(idade_mulheres))
z_stat, p_val = stats.mannwhitneyu(idade_homens, idade_mulheres)  

print("O p-value do teste Mann-Whitney-Wilcoxon entre os Dados = ", p_val )
idade_classe1 = df_train[df_train['Pclass']==1]['Age'].values

idade_classe2 = df_train[df_train['Pclass']==2]['Age'].values

idade_classe3 = df_train[df_train['Pclass']==3]['Age'].values