import pandas as pd



import sklearn

from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns



import numpy as np



%config InlineBackend.figure_format = 'svg' # Graphics in SVG format are more sharp and legible

mpl.rcParams['figure.dpi'] = 300

plt.rcParams['font.size'] = 12



adult_train = pd.read_csv("../input/ucirepo/adult_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
adult_train.head()
adult_train.shape
adult_train.columns
adult_train.info()
adult_train.describe()
adult_train.describe(include=['object'])
adult_train["Education"].value_counts()
adult_train["Education"].value_counts().plot(kind = "bar")
adult_train["Workclass"].value_counts().plot(kind = "bar")
adult_train["Marital Status"].value_counts().plot(kind = "bar")
adult_train["Occupation"].value_counts().plot(kind = "bar")
adult_train["Relationship"].value_counts().plot(kind = "bar")
adult_train["Race"].value_counts().plot(kind = "bar")
adult_train["Sex"].value_counts().plot(kind = "bar")
adult_train["Target"].value_counts().plot(kind = 'bar');
adult_train["Age"].plot(kind = 'hist', bins = 73);
adult_train["Education-Num"].plot(kind = 'hist', bins = 15);
adult_train["Capital Gain"].plot(kind = 'hist', bins = 50);
adult_train["Capital Loss"].plot(kind = 'hist', bins = 50);
adult_train["Hours per week"].plot(kind = 'hist');
df = adult_train.groupby(["Target","Sex"]).mean()

df['Age'].plot(kind = "bar")

plt.ylabel('Age');
df = adult_train.groupby(["Target","Sex"]).mean()

df['Hours per week'].plot(kind = "bar")

plt.ylabel('Hours per week');
df = adult_train.groupby(["Target","Sex"]).mean()

df['Education-Num'].plot(kind = "bar")

plt.ylabel('Education-Num');
df = adult_train.groupby(["Target","Sex"]).mean()

df['Capital Gain'].plot(kind="bar")

plt.ylabel('Capital Gain');
df = adult_train.groupby(["Target","Sex"]).mean()

df['Capital Loss'].plot(kind = "bar")

plt.ylabel('Capital Loss');
pd.crosstab(adult_train["Target"], adult_train["Sex"])
pd.crosstab(adult_train["Target"], adult_train["Education"])
pd.crosstab(adult_train["Target"], adult_train["Workclass"])
pd.crosstab(adult_train["Target"], adult_train["Country"])
train_data = adult_train.dropna()
train_data.shape
train_data
train_data_analysis = train_data

train_data_analysis = train_data_analysis.apply(preprocessing.LabelEncoder().fit_transform)
train_data_analysis.corr()
train_data_analysis.corr().Target.sort_values()
abs(train_data_analysis.corr().Target).sort_values(ascending=False)
test_data = pd.read_csv("../input/ucirepo/adult_test.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?").dropna()
test_data.shape
test_data.head()
x_train = train_data[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

y_train = train_data.Target

x_test = test_data[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

y_test = test_data.Target
knn = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
y_predict
accuracy_score(y_test,y_predict)
knn = KNeighborsClassifier(n_neighbors = 30)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
knn = KNeighborsClassifier(n_neighbors = 16, p = 1)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
k_range = list(range(1, 31))

weight_options = ['uniform', 'distance']

p_options = list(range(1,3))

param_grid = dict(n_neighbors=k_range, p=p_options)
knn = KNeighborsClassifier(n_neighbors=1)



grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)
grid.fit(x_train, y_train)

print(grid.best_estimator_)

print(grid.best_score_)
knn = KNeighborsClassifier(n_neighbors = 18, p = 1)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
scaler = MinMaxScaler()



x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
k_range = list(range(1, 41))

weight_options = ['uniform', 'distance']

p_options = list(range(1,3))

param_grid = dict(n_neighbors=k_range, p=p_options)
knn = KNeighborsClassifier(n_neighbors=1)



grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)
grid.fit(x_train, y_train)

print(grid.best_estimator_)

print(grid.best_score_)
knn = KNeighborsClassifier(n_neighbors = 36, p = 2)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
x_train = train_data[["Capital Gain", "Education-Num", "Relationship", "Age", "Hours per week", "Sex", "Marital Status", 

                      "Capital Loss"]].apply(preprocessing.LabelEncoder().fit_transform)

y_train = train_data.Target

x_test = test_data[["Capital Gain", "Education-Num", "Relationship", "Age", "Hours per week", "Sex", "Marital Status", 

                      "Capital Loss"]].apply(preprocessing.LabelEncoder().fit_transform)

y_test = test_data.Target
knn = KNeighborsClassifier(n_neighbors = 39)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
k_range = list(range(1, 41))

weight_options = ['uniform', 'distance']

p_options = list(range(1,3))

param_grid = dict(n_neighbors=k_range, p=p_options)
knn = KNeighborsClassifier(n_neighbors=1)



grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)
grid.fit(x_train, y_train)

print(grid.best_estimator_)

print(grid.best_score_)
knn = KNeighborsClassifier(n_neighbors = 23, p = 1)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
scaler = MinMaxScaler()



x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
k_range = list(range(1, 41))

weight_options = ['uniform', 'distance']

p_options = list(range(1,3))

param_grid = dict(n_neighbors=k_range, p=p_options)
knn = KNeighborsClassifier(n_neighbors=1)



grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs = -2)
grid.fit(x_train, y_train)

print(grid.best_estimator_)

print(grid.best_score_)
knn = KNeighborsClassifier(n_neighbors = 23, p = 1)

scores = cross_val_score(knn, x_train, y_train, cv = 10)
scores
knn.fit(x_train,y_train)
y_predict = knn.predict(x_test)
accuracy_score(y_test,y_predict)
df_pred_out = pd.DataFrame({'income':y_predict})
df_pred_out.head()
df_pred_out.to_csv("submission.csv", index = True, index_label = 'Id')
pd.options.display.max_columns = 150



household_train = pd.read_csv('../input/householddata/train.csv')

household_test = pd.read_csv('../input/householddata/test.csv')



household_train.head()
household_train.info()
household_train.select_dtypes(np.int64).nunique().value_counts().sort_index().plot.bar(color = 'blue', 

                                                                             figsize = (8, 6));

                                                                            

plt.xlabel('Número de Unique Values'); plt.ylabel('Contagem');

plt.title('Contagem de unique values em colunas de inteiros');
from collections import OrderedDict



plt.figure(figsize = (20, 16))



  

colors = OrderedDict({1: 'red', 2: 'orange', 3: 'blue', 4: 'green'})  # Mapeamento das cores

poverty_mapping = OrderedDict({1: 'extreme', 2: 'moderate', 3: 'vulnerable', 4: 'non vulnerable'})  # Mapeamento níveis pobreza



  # Iterar pelas colunas com variáveis tipo float

for i, col in enumerate(household_train.select_dtypes('float')):

    ax = plt.subplot(4, 2, i+1)

    # Iterate through the poverty levels

    for poverty_level, color in colors.items():

        # Plot each poverty level as a separate line

        sns.kdeplot(household_train.loc[household_train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'Distribuição de {col.capitalize()}'); plt.xlabel(f'{col}'); plt.ylabel('Densidade')



plt.subplots_adjust(top = 2)
household_train.select_dtypes('object').head()
mapping = {"yes": 1, "no": 0}  # Mapeamento para conversão das variáveis





for df in [household_train, household_test]:  # Aplicar a mesma operação para bases de treino e de teste

    # Substituir valores de acordo com o mapeamento (conversão para variável do tipo float)

    df['dependency'] = df['dependency'].replace(mapping).astype(np.float64) 

    df['edjefa'] = df['edjefa'].replace(mapping).astype(np.float64)

    df['edjefe'] = df['edjefe'].replace(mapping).astype(np.float64)



household_train[['dependency', 'edjefa', 'edjefe']].describe()
plt.figure(figsize = (12, 8))



# Itera pelas colunas com variáveis de tipo float

for i, col in enumerate(['dependency', 'edjefa', 'edjefe']):

    ax = plt.subplot(3, 1, i + 1)

    # Itera pelos níveis de pobreza

    for poverty_level, color in colors.items():

        # Desenha cada nível de pobreza como uma linha separada

        sns.kdeplot(household_train.loc[household_train['Target'] == poverty_level, col].dropna(), 

                    ax = ax, color = color, label = poverty_mapping[poverty_level])

        

    plt.title(f'Distribuição de {col.capitalize()}'); plt.xlabel(f'{col}'); plt.ylabel('Densidade')



plt.subplots_adjust(top = 3)
household_test['Target'] = np.nan  # Adiciona coluna Target nula à base de teste

data = household_train.append(household_test, ignore_index = True)
# Chefes de família

heads = data.loc[data['parentesco1'] == 1].copy()



# Labels para treino

train_labels = data.loc[(data['Target'].notnull()) & (data['parentesco1'] == 1), ['Target', 'idhogar']]



# Contagem de valores da variável Target

label_counts = train_labels['Target'].value_counts().sort_index()



# Gráfico de barras das ocorrências para cada label

label_counts.plot.bar(figsize = (8, 6), 

                      color = colors.values(),

                      edgecolor = 'k', linewidth = 2)



plt.xlabel('Nível de pobreza'); plt.ylabel('Contagem'); 

plt.xticks([x - 1 for x in poverty_mapping.keys()], 

           list(poverty_mapping.values()), rotation = 60)

plt.title('Análise do nível de pobreza');



label_counts
# Agrupar a família e descobrir o número de unique values

all_equal = household_train.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)



not_equal = all_equal[all_equal != True]  # Famílias em que os valores da variável Target não são iguais

print('Existem {} famílias cujos membros não apresentam o mesmo valor para a variável Target.'.format(len(not_equal)))

household_train[household_train['idhogar'] == not_equal.index[20]][['idhogar', 'parentesco1', 'Target']]
missing = pd.DataFrame(data.isnull().sum()).rename(columns = {0: 'total'})  # Número de dados faltantes em cada coluna



missing['percentual'] = missing['total'] / len(data)  # Percentual de dados faltantes



missing.sort_values('percentual', ascending = False).head(10).drop('Target')
heads["v18q1"].value_counts()
heads.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())
data['v18q1'] = data['v18q1'].fillna(0)
own_variables = [x for x in data if x.startswith('tipo')]  # Variáveis indicadores de status de propriedade da casa





  # Gráfico representativo das variáveis de status de propriedade para paguamentos de aluguel faltantes

data.loc[data['v2a1'].isnull(), own_variables].sum().plot.bar(figsize = (10, 8),

                                                                        color = 'purple',

                                                              edgecolor = 'k', linewidth = 2);

plt.xticks([0, 1, 2, 3, 4],

           ['Owns and Paid Off', 'Owns and Paying', 'Rented', 'Precarious', 'Other'],

          rotation = 60)

plt.title('Status de Propriedade da Casa para Famílias com Pagamentos de Aluguel Faltantes', size = 15);
data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0  # Preencher pagamento de aluguel com zeros para famílias com casa própria
data.loc[data['rez_esc'].notnull()]['age'].describe()
data.loc[data['rez_esc'].isnull()]['age'].describe()
# Se o indivíduo é maior de 19 ou menor de 7 anos e houver dados faltantes, completar com zero

data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0
id_ = ['Id', 'idhogar', 'Target']
ind_bool = ['v18q', 'dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 

            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 

            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 

            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 

            'parentesco11', 'parentesco12', 'instlevel1', 'instlevel2', 'instlevel3', 

            'instlevel4', 'instlevel5', 'instlevel6', 'instlevel7', 'instlevel8', 

            'instlevel9', 'mobilephone']



ind_ordered = ['rez_esc', 'escolari', 'age']
hh_bool = ['hacdor', 'hacapo', 'v14a', 'refrig', 'paredblolad', 'paredzocalo', 

           'paredpreb','pisocemento', 'pareddes', 'paredmad',

           'paredzinc', 'paredfibras', 'paredother', 'pisomoscer', 'pisoother', 

           'pisonatur', 'pisonotiene', 'pisomadera',

           'techozinc', 'techoentrepiso', 'techocane', 'techootro', 'cielorazo', 

           'abastaguadentro', 'abastaguafuera', 'abastaguano',

            'public', 'planpri', 'noelec', 'coopele', 'sanitario1', 

           'sanitario2', 'sanitario3', 'sanitario5',   'sanitario6',

           'energcocinar1', 'energcocinar2', 'energcocinar3', 'energcocinar4', 

           'elimbasu1', 'elimbasu2', 'elimbasu3', 'elimbasu4', 

           'elimbasu5', 'elimbasu6', 'epared1', 'epared2', 'epared3',

           'etecho1', 'etecho2', 'etecho3', 'eviv1', 'eviv2', 'eviv3', 

           'tipovivi1', 'tipovivi2', 'tipovivi3', 'tipovivi4', 'tipovivi5', 

           'computer', 'television', 'lugar1', 'lugar2', 'lugar3',

           'lugar4', 'lugar5', 'lugar6', 'area1', 'area2']



hh_ordered = [ 'rooms', 'r4h1', 'r4h2', 'r4h3', 'r4m1','r4m2','r4m3', 'r4t1',  'r4t2', 

              'r4t3', 'v18q1', 'tamhog','tamviv','hhsize','hogar_nin',

              'hogar_adul','hogar_mayor','hogar_total',  'bedrooms', 'qmobilephone']



hh_cont = ['v2a1', 'dependency', 'edjefe', 'edjefa', 'meaneduc', 'overcrowding']
sqr_ = ['SQBescolari', 'SQBage', 'SQBhogar_total', 'SQBedjefe', 

        'SQBhogar_nin', 'SQBovercrowding', 'SQBdependency', 'SQBmeaned', 'agesq']
x = ind_bool + ind_ordered + id_ + hh_bool + hh_ordered + hh_cont + sqr_



from collections import Counter



print('Não há repetições: ', np.all(np.array(list(Counter(x).values())) == 1))

print('Cobrimos todas as variáveis: ', len(x) == data.shape[1])
data = data.drop(columns = sqr_)

data.shape
heads = data.loc[data['parentesco1'] == 1, :]

heads = heads[id_ + hh_bool + hh_cont + hh_ordered]

heads.shape
  # Create correlation matrix

corr_matrix = heads.corr()  # Cria matriz de correlação



  # Seleciona triângulo superior da matriz de correlação

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



  # Encontra índice das colunas de features com correlaçã maior que 0.95

to_drop = [column for column in upper.columns if any(abs(upper[column]) > 0.95)]



to_drop
corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9]
sns.heatmap(corr_matrix.loc[corr_matrix['tamhog'].abs() > 0.9, corr_matrix['tamhog'].abs() > 0.9],

            annot=True, cmap = plt.cm.autumn_r, fmt='.3f');
heads = heads.drop(columns = ['tamhog', 'hogar_total', 'r4t3'])
heads = heads.drop(columns = 'area2')



heads.groupby('area1')['Target'].value_counts(normalize = True)
# Usando apenas a base de treino

train_heads = heads.loc[heads['Target'].notnull(), :].copy()



pcorrs = pd.DataFrame(train_heads.corr()['Target'].sort_values()).rename(columns = {'Target': 'pcorr'}).reset_index()

pcorrs = pcorrs.rename(columns = {'index': 'feature'})



print('Variáveis mais negativamente correlacionadas:')

print(pcorrs.head())



print('\nVariáveis mais positivamente correlacionadas:')

print(pcorrs.dropna().tail())
heads.shape
  # Labels para treino

train_labels = np.array(list(heads[heads['Target'].notnull()]['Target'].astype(np.uint8)))



  # Conjuntos de treino e de teste

train_set = heads[heads['Target'].notnull()].drop(columns = ['Id', 'idhogar', 'Target'])

test_set = heads[heads['Target'].isnull()].drop(columns = ['Id', 'idhogar', 'Target'])



  # Base de submissão

submission_base = household_test[['Id', 'idhogar']].copy()
train_set = train_set.apply(preprocessing.LabelEncoder().fit_transform)

y_train_set = heads.Target.dropna()
knn = KNeighborsClassifier(n_neighbors = 3)

scores = cross_val_score(knn, train_set, y_train_set, cv = 10)
scores