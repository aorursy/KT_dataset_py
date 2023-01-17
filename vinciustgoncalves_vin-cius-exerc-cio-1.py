import pandas as pd

df = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

df.head(1)
resposta = [["aeronave_operador_categoria", "Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_modelo","Qualitativa Nominal"],

            ["aeronave_motor_tipo","Quantitativa Nominal"],

            ["aeronave_ano_fabricacao","Quantitativa Discreta"], 

            ["aeronave_fase_operacao","Qualitativa Discreta"], 

            ["total_fatalidades","Quantitativa Discreta"]]



resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
aeronave_operador_categoria = pd.DataFrame(df["aeronave_operador_categoria"].value_counts())

aeronave_operador_categoria
aeronave_tipo_veiculo = pd.DataFrame(df["aeronave_tipo_veiculo"].value_counts())

aeronave_tipo_veiculo
aeronave_modelo = pd.DataFrame(df["aeronave_modelo"].value_counts())

aeronave_modelo
aeronave_motor_tipo = pd.DataFrame(df["aeronave_motor_tipo"].value_counts())

aeronave_motor_tipo
import matplotlib.pyplot as plt
aeronave_operador_categoria.sort_values("aeronave_operador_categoria", ascending=False).plot.bar()
tipo = df['aeronave_tipo_veiculo'].value_counts().reset_index().head(10)



plt.figure(figsize=(6, 6))

plt.barh(y=tipo['index'], width=tipo['aeronave_tipo_veiculo'])

plt.title('Tipo Veiculo')

plt.xlabel('Quantidade')

plt.ylabel('Tipo')



plt.show()
modelo = df["aeronave_modelo"].value_counts().reset_index().head(10)



plt.figure(figsize=(15, 8))

plt.bar(modelo['index'], modelo["aeronave_modelo"], color='blue')

plt.title('Modelos Aeronaves')

plt.xlabel('Modelo')

plt.ylabel('Quantidade')

plt.show()
motor_tipo = df["aeronave_motor_tipo"].value_counts().reset_index().head(10)



plt.figure(figsize=(15, 8))

plt.bar(motor_tipo['index'], motor_tipo["aeronave_motor_tipo"], color='green')

plt.title('Tipo do Motor')

plt.xlabel('Motor')

plt.ylabel('Quantidade')

plt.show()
fase = df['aeronave_fase_operacao'].value_counts().reset_index().head(10)



plt.figure(figsize=(6, 6))

plt.barh(y=fase['index'], width=fase['aeronave_fase_operacao'], color="orange")

plt.title('Tipo Veiculo')

plt.xlabel('Quantidade')

plt.ylabel('Tipo')



plt.show()