import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv("../input/2016.csv")
df.head()
plt.style.use("ggplot")
plt.figure(figsize = (20, 12))
df["enem_nt_mt"].hist(bins = 40, ec = "k", alpha = .6, color = "royalblue")
plt.title("Distribuição das notas em matemática")
plt.xlabel("Notas")
plt.ylabel("Contagem")
plt.figure(figsize = (20, 12))
df[df["enem_nt_mt"] > 900]["enem_nt_mt"].hist(bins = 20, ec = "k", alpha = .6, color = "royalblue")
plt.title("Distribuição de notas em matemática acima de 900 (ENEM 2016)")
plt.xlabel("Notas")
plt.ylabel("Contagem")
count = df["nu_ano_inicio_graduacao"].value_counts().values
anos = df["nu_ano_inicio_graduacao"].value_counts().index
plt.figure(figsize = (24, 8))
plt.subplot(1, 2, 1)
plt.bar(anos, count, ec = "k", alpha = .6, color = "royalblue")
plt.xlabel("Ano")
plt.title("Ano de início de graduação dos alunos que realizaram o ENADE em 2016")
plt.subplot(1, 2, 2)
plt.pie(count, 
        labels = list(anos),  
        colors = ["#20257c", "#424ad1", "#6a8ee8", "#66bbe2", "#66dee2", "#6ce2cb", "#6ad187", "#3b7f5b"],
        labeldistance = 1.1,
        explode = [0, 0, 0, .1, .2, .4, .6, .8],
        wedgeprops = {"ec": "k"}, 
        textprops = {"fontsize": 15}, 
        )
plt.axis("equal")
plt.title("Ano de início de graduação dos alunos que realizaram o ENADE em 2016")
plt.legend()
indexes = df[df["enem_nt_mt"] == 0].index
df = df.drop(indexes)
indexes = df[df["enem_nt_ch"] == 0].index
df = df.drop(indexes)
notas_max = [df["enem_nt_cn"].max(), df["enem_nt_ch"].max(), df["enem_nt_lc"].max(), df["enem_nt_mt"].max()]
notas_min = [df["enem_nt_cn"].min(), df["enem_nt_ch"].min(), df["enem_nt_lc"].min(), df["enem_nt_mt"].min()]
plt.figure(figsize = (20, 14))
bar_width = .35
index = np.arange(4)

plt.barh(index, 
         notas_max, 
         ec = "k", 
         alpha = .6, 
         color = "royalblue", 
         height = bar_width, 
         label = "Máxima")

plt.barh(index - bar_width, 
         notas_min, 
         ec = "k", 
         alpha = .6, 
         color = "darkblue", 
         height = bar_width, 
         label = "Mínima")

for i, v in enumerate(notas_max):
    plt.text(v - 50, i, str(v))
for i, v in enumerate(notas_min):
    plt.text(v - 50, i - bar_width, str(v))
        
plt.yticks(index - bar_width / 2, ("Ciências Naturais", "Ciências Humanas", "Linguagem e Códigos", "Matemática"))
plt.title("Notas máximas e mínimas em cada área (ENEM 2016)")
plt.legend()