import pandas as pd
resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)
resposta = [["total_fatalidades", "Quantitativa Discreta"],["aeronave_tipo_veiculo","Variável categórica"], ["aeronave_matricula","Variável categórica"], ["aeronave_ano_fabricacao","Quantitativa Contínua"], ["aeronave_fabricante","Qualitativa Nominal"],  ["aeronave_modelo","Qualitativa Nominal"], ["aeronave_assentos","Quantitativa Discreta"], ["aeronave_pmd_categoria","Qualitativa Ordinal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])
resposta
freq_aeronave_fabricante = df['aeronave_fabricante'].value_counts()
freq_aeronave_fabricante
freq_aeronave_fabricante = pd.DataFrame(freq_aeronave_fabricante).unstack().reset_index(name='Freq')
freq_aeronave_fabricante.drop('level_0', axis=1, inplace=True)
freq_aeronave_fabricante.columns = ['Aeronave', 'Frequencia absoluta']
freq_aeronave_fabricante

freqrel_aeronave_fabricante = [round(df['aeronave_fabricante'].value_counts()/df.shape[0]*100,2)]
freqrel_aeronave_fabricante = pd.DataFrame(freqrel_aeronave_fabricante).unstack().reset_index(name='Freq')
freqrel_aeronave_fabricante.drop('level_1', axis=1, inplace=True)
freqrel_aeronave_fabricante.columns = ['Aeronave', 'Frequencia Relativa (%)']
mergedFreq = pd.merge(freq_aeronave_fabricante, freqrel_aeronave_fabricante )
mergedFreq
freq_aeronave_fabricante = df['aeronave_modelo'].value_counts()
freq_aeronave_fabricante
freq_aeronave_fabricante = pd.DataFrame(freq_aeronave_fabricante).unstack().reset_index(name='Freq')
freq_aeronave_fabricante.drop('level_0', axis=1, inplace=True)
freq_aeronave_fabricante.columns = ['Aeronave', 'Frequencia absoluta']
freq_aeronave_fabricante

freqrel_aeronave_fabricante = [round(df['aeronave_modelo'].value_counts()/df.shape[0]*100,2)]
freqrel_aeronave_fabricante = pd.DataFrame(freqrel_aeronave_fabricante).unstack().reset_index(name='Freq')
freqrel_aeronave_fabricante.drop('level_1', axis=1, inplace=True)
freqrel_aeronave_fabricante.columns = ['Aeronave', 'Frequencia Relativa (%)']
mergedFreq = pd.merge(freq_aeronave_fabricante, freqrel_aeronave_fabricante )
mergedFreq
freq_aeronave_fabricante = df['aeronave_pmd_categoria'].value_counts()
freq_aeronave_fabricante = pd.DataFrame(freq_aeronave_fabricante).unstack().reset_index(name='Freq')
freq_aeronave_fabricante.drop('level_0', axis=1, inplace=True)
freq_aeronave_fabricante.columns = ['Aeronave', 'Frequencia absoluta']
freq_aeronave_fabricante

freqrel_aeronave_fabricante = [round(df['aeronave_pmd_categoria'].value_counts()/df.shape[0]*100,2)]
freqrel_aeronave_fabricante = pd.DataFrame(freqrel_aeronave_fabricante).unstack().reset_index(name='Freq')
freqrel_aeronave_fabricante.drop('level_1', axis=1, inplace=True)
freqrel_aeronave_fabricante.columns = ['Aeronave', 'Frequencia Relativa (%)']
mergedFreq = pd.merge(freq_aeronave_fabricante, freqrel_aeronave_fabricante )
mergedFreq
freq_aeronave_modelo = df['aeronave_modelo'].value_counts()
freq_aeronave_pmd_categoria = df['aeronave_pmd_categoria'].value_counts()
import matplotlib.pyplot as plt
%matplotlib inline

tipo_veiculo = df['aeronave_tipo_veiculo']
total_fatalidade = df['total_fatalidades']

plt.title('Total mortos por Tipo veiculo')
plt.xlabel('Tipo de Veiculo')
plt.ylabel('Quantidade de Fatalidades')
plt.bar(tipo_veiculo,total_fatalidade)
plt.xticks(rotation='vertical')
plt.show()
aeronave_categoria = df["aeronave_pmd_categoria"].value_counts()
aeronave_categoria = pd.DataFrame(aeronave_categoria).unstack().reset_index(name='Freq')
aeronave_categoria.drop('level_0', axis=1, inplace=True)
plt.title('Porcentagem de acidentes por peso de aeronaves')

explode = (0.1, 0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
plt.pie(aeronave_categoria['Freq'], explode=explode, labels=aeronave_categoria['level_1'], shadow=True, autopct='%1.1f%%')
plt.show()
aeronave_fabricante = df["aeronave_fabricante"].value_counts()
aeronave_fabricante = pd.DataFrame(aeronave_fabricante.head(10)).unstack().reset_index(name='Freq')
aeronave_fabricante.drop('level_0', axis=1, inplace=True)

plt.title('Quantidade de acidentes por fabricante')
plt.xlabel('Fabricante')
plt.ylabel('Quantidade de Fatalidades')
plt.bar(aeronave_fabricante['level_1'],aeronave_fabricante['Freq'])
plt.xticks(rotation='vertical')
plt.show()

aeronave_motor_tipo = df["aeronave_motor_tipo"].value_counts()
aeronave_motor_tipo = pd.DataFrame(aeronave_motor_tipo).unstack().reset_index(name='Freq')
aeronave_motor_tipo.drop('level_0', axis=1, inplace=True)
aeronave_motor_tipo

plt.title('Quantidade de acidentes por tipo de motor')
plt.xlabel('Tipo de motor')
plt.ylabel('Quantidade de Fatalidades')
plt.bar(aeronave_motor_tipo['level_1'],aeronave_motor_tipo['Freq'])
plt.xticks(rotation=45)
plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)