#carregando pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#carregando dados
dados = pd.read_csv("../input/amostra-enem-2019/MICRODADOS_ENEM_2019_SAMPLE_43278.csv")
dados.head()
dados.shape
#Essa amostra possui 127380 alunos, de um espaço amostral de cerca de 5 milhões de participantes da prova.
print(dados["SG_UF_RESIDENCIA"].unique())
len(dados["SG_UF_RESIDENCIA"].unique())
dados["SG_UF_RESIDENCIA"].value_counts()
dados["NU_IDADE"].value_counts()
prop = dados["NU_IDADE"].value_counts()/len(dados)
prop
dados["NU_IDADE"].value_counts().sort_index()
dados.query("NU_IDADE == 13")["SG_UF_RESIDENCIA"].value_counts().sort_index()
dados["NU_IDADE"].hist(bins = 40, figsize = (12,10)).set_title("Idade dos participantes do ENEM 2019")
dados.query("IN_TREINEIRO == 1")["NU_IDADE"].value_counts().sort_index()

fig = plt.figure(figsize=(12, 10))
ax0 = fig.add_axes([0.1, 0.6, 0.8, 0.4])
ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.4])

ax0.hist(dados.query("IN_TREINEIRO == 1")["NU_IDADE"], bins= 50)
ax0.set_title("Treineiros")
ax1.hist(dados.query("IN_TREINEIRO == 0")["NU_IDADE"], bins = 50)
ax1.set_title("Não Treineiros")
provas = ["NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_MT","NU_NOTA_LC","NU_NOTA_REDACAO"]

dados[provas].describe()
dados[provas].boxplot(grid=True, figsize= (10,8))
dados.query("TP_LINGUA == 0")["NU_NOTA_LC"].plot.box(grid = True, figsize=(8,6))
_ = plt.title("Grafico de caixa das notas de linguagem e códigos para quem escolheu inglês")
dados.query("TP_LINGUA == 1")["NU_NOTA_LC"].plot.box(grid = True, figsize=(8,6))
_ = plt.title("Grafico de caixa das notas de linguagem e códigos para quem escolheu espanhol")
idiomas = ['Espanhol','Ingles']

ing = dados.query("TP_LINGUA == 0")['NU_NOTA_LC']
esp = dados.query("TP_LINGUA == 1")['NU_NOTA_LC']


dados_por_idioma = pd.concat([esp, ing], axis=1, names = idiomas)
dados_por_idioma.columns = idiomas
dados_por_idioma.plot.box(grid=True, figsize=(8, 6))
_ = plt.title("Notas da prova de linguagem e códigos")
dados.query("NU_IDADE <= 17")["SG_UF_RESIDENCIA"].value_counts()
dados.query("NU_IDADE <= 17")["SG_UF_RESIDENCIA"].value_counts(normalize=True)
alunos_menor_18 = dados.query("NU_IDADE <= 17")
alunos_menor_18["SG_UF_RESIDENCIA"].value_counts().plot.pie(figsize=(10,8))
alunos_menor_18["SG_UF_RESIDENCIA"].value_counts(normalize = True).plot.bar(figsize=(10,8))
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.boxplot(x="Q006", y = "NU_NOTA_MT", data = dados)
plt.title("Boxplot das notas de matemática pela renda")
renda_ordenada = dados["Q006"].unique()
renda_ordenada.sort()
plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_MT", data = dados, order = renda_ordenada)
plt.title("Boxplot das notas de matemática pela renda")
dados["NU_NOTA_TOTAL"] = dados[provas].sum(axis=1)

plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados, order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")
provas = ["NU_NOTA_CN","NU_NOTA_CH","NU_NOTA_MT","NU_NOTA_LC","NU_NOTA_REDACAO"]
provas.append("NU_NOTA_TOTAL")
dados[provas].query("NU_NOTA_TOTAL == 0")
dados_sem_notas_zero = dados.query("NU_NOTA_TOTAL != 0")

plt.figure(figsize=(10, 6))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados_sem_notas_zero, order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")

plt.figure(figsize=(14, 8))
sns.boxplot(x="Q006", y = "NU_NOTA_TOTAL", data = dados_sem_notas_zero, 
            hue = "IN_TREINEIRO", order = renda_ordenada)
plt.title("Boxplot das notas de total pela renda")