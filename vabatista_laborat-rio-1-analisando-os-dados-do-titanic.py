%matplotlib inline
import pandas as pd
import numpy as np
import pylab as plt

# define tamanho padrão das figuras
plt.rc('figure', figsize=(10, 5))
fizsize_with_subplots = (10, 10)
bin_size = 10 

# df_train é um dataframe pandas.
df_train = pd.read_csv('../input/lab1_train.csv')
df_train.head(5)
df_train.info()
df_train.describe()
fig = plt.figure(figsize=fizsize_with_subplots) 
fig_dims = (3, 2)

# Mortes x Sobreviventes
plt.subplot2grid(fig_dims, (0, 0))
df_train['Survived'].value_counts().plot(kind='bar', title='Mortos x Sobreviventes')

# Classe do passageiro
plt.subplot2grid(fig_dims, (0, 1))
df_train['Pclass'].value_counts().plot(kind='bar', title='Classe do passageiro')

# Gênero
plt.subplot2grid(fig_dims, (1, 0))
df_train['Sex'].value_counts().plot(kind='bar', title='Genero')
plt.xticks(rotation=0)

# Porto de embarque
plt.subplot2grid(fig_dims, (1, 1))
df_train['Embarked'].value_counts().plot(kind='bar', title='Porto de embarque')

# Idade
plt.subplot2grid(fig_dims, (2, 0))
df_train['Age'].hist()
plt.title('Idade')

# Tarifa
plt.subplot2grid(fig_dims, (2, 1))
df_train['Fare'].hist()
plt.title('Tarifa')
pclass_xt = pd.crosstab(df_train['Pclass'], df_train['Survived'])
pclass_xt
# Normaliza a tabela para a soma dar 1:
pclass_xt_pct = pclass_xt.div(pclass_xt.sum(1).astype(float), axis=0)

pclass_xt_pct.plot(kind='bar', stacked=True, title='Taxa de sobrevivencia por classe do passageiro')
plt.xlabel('Classe do Passageiro')
plt.ylabel('Taxa de sobrevivencia')
sex_val_xt = pd.crosstab(df_train['Sex'], df_train['Survived'])
sex_val_xt_pct = sex_val_xt.div(sex_val_xt.sum(1).astype(float), axis=0)
sex_val_xt_pct.plot(kind='bar', stacked=True, title='Taxa de sobrevivencia por genero')
passenger_classes = df_train['Pclass'].unique()

for p_class in passenger_classes:
    print('M: ', p_class, len(df_train[(df_train['Sex'] == 'male') & 
                             (df_train['Pclass'] == p_class)]))
    print('F: ', p_class, len(df_train[(df_train['Sex'] == 'female') & 
                             (df_train['Pclass'] == p_class)]))
females_df = df_train[df_train['Sex'] == 'female']
females_xt = pd.crosstab(females_df['Pclass'], df_train['Survived'])
females_xt_pct = females_xt.div(females_xt.sum(1).astype(float), axis=0)
females_xt_pct.plot(kind='bar', stacked=True, title='Taxa de sobrevivencia das mulheres por classe')
plt.xlabel('Classe do passageiro')
plt.ylabel('Taxa de sobrevivencia')

males_df = df_train[df_train['Sex'] == 'male']
males_xt = pd.crosstab(males_df['Pclass'], df_train['Survived'])
males_xt_pct = males_xt.div(males_xt.sum(1).astype(float), axis=0)
males_xt_pct.plot(kind='bar', stacked=True, title='Taxa de sobrevivencia dos homens por classe')
plt.xlabel('Classe do passageiro')
plt.ylabel('Taxa de sobrevivencia')

