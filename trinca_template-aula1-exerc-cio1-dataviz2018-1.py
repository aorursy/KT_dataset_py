import pandas as pd
import matplotlib.pyplot as plt

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = [['aeronave_fabricante' , 'Qualitativa Nominal'],['aeronave_operador_categoria' , 'Qualitativa Ordinal'],['aeronave_tipo_veiculo' , 'Qualitativa Nominal'],
            ['aeronave_modelo' , 'Qualitativa Ordinal'],['aeronave_motor_tipo' , 'Qualitativa Nominal'],
            ['aeronave_motor_quantidade' , 'Qualitativa Ordinal'],['aeronave_pais_registro' , 'Qualitativa Nominal'],
            ['total_fatalidades' , 'Quantitativa Discreta']]
resposta = pd.DataFrame(resposta, columns=["Variável", "Classificação"])
resposta
dataset = pd.read_csv('../input/anv.csv', delimiter=',')
fabricante = dataset['aeronave_fabricante'].value_counts()
cFabricante = dataset['aeronave_fabricante'].value_counts(normalize=True)
tb = pd.concat([fabricante, cFabricante], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequência - Aeronaves por Fabricante')
print(tb)

### Variavel aeronave_operador_categoria
categoria = dataset['aeronave_operador_categoria'].value_counts()
cCategoria = dataset['aeronave_operador_categoria'].value_counts(normalize=True)
tb = pd.concat([categoria, cCategoria], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Operador')
print(tb)

### Variavel aeronave_tipo_veiculo
tipo = dataset['aeronave_tipo_veiculo'].value_counts()
cTipo = dataset['aeronave_tipo_veiculo'].value_counts(normalize=True)
tb = pd.concat([tipo, cTipo], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Tipo')
print(tb)

### Variavel aeronave_modelo
modelo = dataset['aeronave_modelo'].value_counts()
cModelo = dataset['aeronave_modelo'].value_counts(normalize=True)
tb = pd.concat([modelo, cModelo], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Modelo')
print(tb)

### Variavel aeronave_motor_tipo
motor = dataset['aeronave_motor_tipo'].value_counts()
cMotor = dataset['aeronave_motor_tipo'].value_counts(normalize=True)
tb = pd.concat([motor, cMotor], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Tipo de Motor')
print(tb)

### Variavel aeronave_motor_quantidade
quantidade = dataset['aeronave_motor_quantidade'].value_counts()
cQuantidade = dataset['aeronave_motor_quantidade'].value_counts(normalize=True)
tb = pd.concat([quantidade, cQuantidade], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Quantidade de Motor')
print(tb)

### Variavel aeronave_pais_registro
registro = dataset['aeronave_pais_registro'].value_counts()
cRegistro = dataset['aeronave_pais_registro'].value_counts(normalize=True)
tb = pd.concat([registro, cRegistro], axis=1, keys=['Absoluta', 'Relativa %'])
print('Frequências - Aeronaves por Pais de Registro')
print(tb)
### Grafico para demonstrar variavel Qualitativa
motor = dataset[dataset.aeronave_motor_quantidade != '***']
cmotor = motor['aeronave_motor_quantidade'].value_counts()
labels = cmotor.keys().tolist()
print(cmotor)

fig1, ax1 = plt.subplots()
ax1.pie(cmotor, explode=(0, 0, 0, 0, 0), autopct='%1.1f%%',shadow=False, startangle=90)
ax1.axis('equal')
ax1.legend(labels = labels,loc="center left",bbox_to_anchor=(1, 0, 0, 1))
ax1.set_title("Quantidade Motores Utilizadaos Por Tipo de Aviões")
plt.show()

### Grafico para demonstrar variavel Quantitativa
fase = dataset[(dataset.aeronave_fase_operacao != '***') & (dataset.aeronave_fase_operacao != 'INDETERMINADA')].groupby(['aeronave_fase_operacao'])['total_fatalidades'].sum()
cfase = fase.sort_values()[fase > 0].sort_values(ascending=False)[0:12].sort_values()
labels = fase.keys().tolist()
print(fase)

plt.rcdefaults()
fig, ax = plt.subplots()
plt.xticks(rotation='vertical')

ax.bar(labels, fase, align='center', ecolor='black', color='#ff4422')
ax.set_title('Fase de operação com maior número de mortes')

plt.show()
df = pd.read_csv('../input/anv.csv', delimiter=',')
df.head(1)