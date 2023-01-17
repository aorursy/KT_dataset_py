import pandas as pd

resposta = [["idade", "Quantitativa Discreta"],["sexo","Qualitativa Nominal"]] #variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
df = pd.read_csv('../input/anv.csv', delimiter=',')

df.head(5)
import matplotlib.pyplot as plt
# Lista de colunas escolhidas e sua classificação:

resposta = [["aeronave_operador_categoria","Qualitativa Nominal"],

            ["aeronave_tipo_veiculo","Qualitativa Nominal"],

            ["aeronave_motor_tipo","Qualitativa Nominal"],

            ["aeronave_motor_quantidade","Qualitativa Nominal"],

            ["aeronave_pmd_categoria","Qualitativa Nominal"],

            ["aeronave_assentos","Quantitativa Discreta"],

            ["total_fatalidades","Quantitativa Discreta"]]

resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificação"])

resposta
# Tabela de Frequência para a coluna "aeronave_operador_categoria"

df["aeronave_operador_categoria"].value_counts()
# Tabela de Frequência para a coluna "aeronave_tipo_veiculo"

df["aeronave_tipo_veiculo"].value_counts()
# Tabela de Frequência para a coluna "aeronave_motor_tipo"

df["aeronave_motor_tipo"].value_counts()
# Tabela de Frequência para a coluna "aeronave_motor_quantidade"

df["aeronave_motor_quantidade"].value_counts()
# Tabela de Frequência para a coluna "aeronave_pmd_categoria"

df["aeronave_pmd_categoria"].value_counts()
# Tabela de Frequência para a coluna "aeronave_assentos"

df["aeronave_assentos"].value_counts()
# Tabela de Frequência para a coluna "total_fatalidades"

df["total_fatalidades"].value_counts()
df_grafico = df["aeronave_operador_categoria"].value_counts().plot(kind='bar')

df_grafico.set_title("aeronave_operador_categoria", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_operador_categoria'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_operador_categoria'])

    plt.title('aeronave_operador_categoria')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["aeronave_tipo_veiculo"].value_counts().plot(kind='bar')

df_grafico.set_title("aeronave_tipo_veiculo", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_tipo_veiculo'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_tipo_veiculo'])

    plt.title('aeronave_tipo_veiculo')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["aeronave_motor_tipo"].value_counts().plot(kind='bar')

df_grafico.set_title("aeronave_motor_tipo", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_motor_tipo'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_motor_tipo'])

    plt.title('aeronave_motor_tipo')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["aeronave_motor_quantidade"].value_counts().plot(kind='bar')

df_grafico.set_title("aeronave_motor_quantidade", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_motor_quantidade'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_motor_quantidade'])

    plt.title('aeronave_motor_quantidade')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["aeronave_pmd_categoria"].value_counts().plot(kind='bar')

df_grafico.set_title("aeronave_pmd_categoria", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_pmd_categoria'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_pmd_categoria'])

    plt.title('aeronave_pmd_categoria')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["aeronave_assentos"].value_counts().head(5).plot(kind='bar')

df_grafico.set_title("aeronave_assentos", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['aeronave_assentos'].value_counts().head(5).reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['aeronave_assentos'])

    plt.title('aeronave_assentos')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()
df_grafico = df["total_fatalidades"].value_counts().head(5).plot(kind='bar')

df_grafico.set_title("total_fatalidades", fontsize=18)

df_grafico.set_ylabel("qtd", fontsize=12);
data = df['total_fatalidades'].value_counts().reset_index().head()



with plt.style.context('default'):

    plt.figure(figsize=(5, 3))

    plt.pie(x=data['total_fatalidades'])

    plt.title('total_fatalidades')

    plt.legend(labels=data['index'], bbox_to_anchor=(1, 1))

    plt.show()