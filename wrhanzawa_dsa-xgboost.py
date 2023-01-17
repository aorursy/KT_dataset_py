import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

import statsmodels.api as sm

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from scipy.stats import ks_2samp

from sklearn.metrics import log_loss

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_validate

from sklearn.metrics import log_loss, make_scorer

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from scipy import stats
df_dsa = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_treino.csv')

df_dsa.head()
var_num = [x for x in df_dsa.columns if df_dsa[x].dtype in ('float64','float32','int64','int32')]
for var in var_num:

    df_dsa[var] = df_dsa[var].fillna(df_dsa[var].median())
df_dsa.shape
df_treino = df_dsa.copy()
var_inicio = [x for x in df_treino.columns if x.startswith('v')]

len(var_inicio)
def concentracao_valores(df, lista_var, pct):

    var_select = []

    for var in lista_var:

        if df[var].value_counts().max()/len(df) < pct:

            var_select.append(var)

    return var_select

var_select_concentracao = concentracao_valores(df_treino, var_inicio, 0.95)
len(var_select_concentracao)
var_num = [x for x in df_treino[var_select_concentracao].columns if df_treino[x].dtype in ('float64','float32','int64','int32')]
def select_corr(df,lista_var, max_corr, target):

    import math

    tabela_correlacao = df[var_select_concentracao].corr()

    tabela_correlacao_ok = pd.DataFrame(tabela_correlacao).stack().reset_index()

    tabela_correlacao_ok.columns = ['VAR1','VAR2','CORR']

    tabela_correlacao_ok = tabela_correlacao_ok[tabela_correlacao_ok['VAR1'] != tabela_correlacao_ok['VAR2']]

    dict_corr = {'var':[],'corr_target':[]}

    for var in lista_var:

        dict_corr['var'].append(var)

        dict_corr['corr_target'].append(abs(stats.pointbiserialr(df_treino[target],df_treino[var])[0]))

        df_corr= pd.DataFrame(dict_corr)

    tabela_correlacao_ok = tabela_correlacao_ok.merge(df_corr,

                                                      how = 'left',

                                                      left_on = 'VAR1', 

                                                      right_on='var').merge(df_corr,

                                                                            how = 'left',

                                                                            left_on = 'VAR2', 

                                                                            right_on='var', suffixes = ['_1','_2'])

        

    tabela_correlacao_ok['CORR'] = abs(tabela_correlacao_ok['CORR'])

    tabela_correlacao_ok.sort_values('CORR', ascending = False, inplace = True)

    lista_var_drop = []

    while tabela_correlacao_ok['CORR'].max() >= max_corr:

        if tabela_correlacao_ok.iloc[0]['corr_target_2'] > tabela_correlacao_ok.iloc[0]['corr_target_1']:

            var_drop = tabela_correlacao_ok.iloc[0]['VAR1']

        else:

            var_drop = tabela_correlacao_ok.iloc[0]['VAR2']

        lista_var_drop.append(var_drop)

        tabela_correlacao_ok = tabela_correlacao_ok[tabela_correlacao_ok['VAR1'] != var_drop]

        tabela_correlacao_ok = tabela_correlacao_ok[tabela_correlacao_ok['VAR2'] != var_drop]

    return lista_var_drop

lista_var_drop = select_corr(df_treino, var_num, 0.7, 'target')
lista_var_ok = [x for x in var_select_concentracao if x not in lista_var_drop]
var_categ = [x for x in var_select_concentracao if df_dsa[x].dtype not in ('float64','float32','int64','int32')]
df_treino[var_categ] = df_treino[var_categ].fillna(-99)
len(lista_var_ok)
var_categ = [x for x in lista_var_ok if df_treino[x].dtype not in ('float64','float32','int64','int32')]
var_categ
df_treino.loc[df_treino['v24'] == 'A', 'CT_v24'] = 1.0

df_treino.loc[df_treino['v24'] == 'B', 'CT_v24'] = 2.0

df_treino.loc[df_treino['v24'] == 'C', 'CT_v24'] = 3.0

df_treino.loc[df_treino['v24'] == 'D', 'CT_v24'] = 1.0

df_treino.loc[df_treino['v24'] == 'E', 'CT_v24'] = 2.0



df_treino.loc[df_treino['v30'] == -99, 'CT_v30'] = 2.0

df_treino.loc[df_treino['v30'] == 'A', 'CT_v30'] = 3.0

df_treino.loc[df_treino['v30'] == 'B', 'CT_v30'] = 3.0

df_treino.loc[df_treino['v30'] == 'C', 'CT_v30'] = 2.0

df_treino.loc[df_treino['v30'] == 'D', 'CT_v30'] = 1.0

df_treino.loc[df_treino['v30'] == 'E', 'CT_v30'] = 1.0

df_treino.loc[df_treino['v30'] == 'F', 'CT_v30'] = 3.0

df_treino.loc[df_treino['v30'] == 'G', 'CT_v30'] = 1.0



df_treino.loc[df_treino['v31'] == -99, 'CT_v31'] = 2.0

df_treino.loc[df_treino['v31'] == 'A', 'CT_v31'] = 1.0

df_treino.loc[df_treino['v31'] == 'B', 'CT_v31'] = 2.0

df_treino.loc[df_treino['v31'] == 'C', 'CT_v31'] = 2.0



df_treino.loc[df_treino['v47'] == 'A', 'CT_v47'] = 2.0

df_treino.loc[df_treino['v47'] == 'B', 'CT_v47'] = 2.0

df_treino.loc[df_treino['v47'] == 'C', 'CT_v47'] = 2.0

df_treino.loc[df_treino['v47'] == 'D', 'CT_v47'] = 2.0

df_treino.loc[df_treino['v47'] == 'E', 'CT_v47'] = 1.0

df_treino.loc[df_treino['v47'] == 'F', 'CT_v47'] = 1.0

df_treino.loc[df_treino['v47'] == 'G', 'CT_v47'] = 1.0

df_treino.loc[df_treino['v47'] == 'H', 'CT_v47'] = 1.0

df_treino.loc[df_treino['v47'] == 'I', 'CT_v47'] = 1.0

df_treino.loc[df_treino['v47'] == 'J', 'CT_v47'] = 1.0



df_treino['CT_v56'] = 3.0

df_treino.loc[df_treino['v56'] == -99 ,'CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'CN','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'DF','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'DO','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'DY','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'G','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'P','CT_v56'] = 1.0

df_treino.loc[df_treino['v56'] == 'BL','CT_v56'] = 2.0

df_treino.loc[df_treino['v56'] == 'BZ','CT_v56'] = 2.0

df_treino.loc[df_treino['v56'] == 'DX','CT_v56'] = 2.0

df_treino.loc[df_treino['v56'] == 'AG','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'AW','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'BJ','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'BK','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'DA','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'DH','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'DI','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'N','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'R','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'U','CT_v56'] = 4.0

df_treino.loc[df_treino['v56'] == 'AL','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'BM','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'BQ','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'CS','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'CY','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'DR','CT_v56'] = 5.0

df_treino.loc[df_treino['v56'] == 'V','CT_v56'] = 5.0





df_treino['CT_v66'] = df_treino['v66']



df_treino['CT_v71'] = 2.0

df_treino.loc[(df_treino['v71'] == 'C') , 'CT_v71'] = 1.0



df_treino.loc[(df_treino['v79'] == 'A') | 

              (df_treino['v79'] == 'B') | 

              (df_treino['v79'] == 'F') |

              (df_treino['v79'] == 'N') |

              (df_treino['v79'] == 'O') |

              (df_treino['v79'] == 'R') |

              (df_treino['v79'] == 'J') , 'CT_v79'] = 3.0

df_treino.loc[(df_treino['v79'] == 'D') | 

              (df_treino['v79'] == 'M') |

              (df_treino['v79'] == 'Q') |

              (df_treino['v79'] == 'C') |

              (df_treino['v79'] == 'H') |

              (df_treino['v79'] == 'K') |

              (df_treino['v79'] == 'G') , 'CT_v79'] = 1.0

df_treino.loc[(df_treino['v79'] == 'E') | 

              (df_treino['v79'] == 'I') |

              (df_treino['v79'] == 'P') , 'CT_v79'] = 2.0



df_treino['CT_v110'] = 2.0

df_treino.loc[(df_treino['v110'] == 'A'), 'CT_v110'] = 1.0



df_treino['CT_v113'] = 1

df_treino.loc[(df_treino['v113'] == -99) | 

              (df_treino['v113'] == 'AD') |

              (df_treino['v113'] == 'C') |

              (df_treino['v113'] == 'E') |

              (df_treino['v113'] == 'F') |

              (df_treino['v113'] == 'H') |

              (df_treino['v113'] == 'M') |

              (df_treino['v113'] == 'N') |

              (df_treino['v113'] == 'O') |

              (df_treino['v113'] == 'T') , 'CT_v113'] = 2
lista_var_ok.remove('v22')

lista_var_ok.remove('v52')

lista_var_ok.remove('v56')

lista_var_ok.remove('v75')

lista_var_ok.remove('v91')

lista_var_ok.remove('v107')

lista_var_ok.remove('v112')

lista_var_ok.remove('v125')

lista_var_ok.remove('v66')

lista_var_ok.remove('v24')

lista_var_ok.remove('v30')

lista_var_ok.remove('v31')

lista_var_ok.remove('v47')

lista_var_ok.remove('v71')

lista_var_ok.remove('v79')

lista_var_ok.remove('v110')

lista_var_ok.remove('v113')

lista_var_ok.append('CT_v24')

lista_var_ok.append('CT_v30')

lista_var_ok.append('CT_v31')

lista_var_ok.append('CT_v47')

lista_var_ok.append('CT_v66')

lista_var_ok.append('CT_v71')

lista_var_ok.append('CT_v79')

lista_var_ok.append('CT_v110')

lista_var_ok.append('CT_v113')

lista_var_ok.append('CT_v56')
len(lista_var_ok)
var_num_fim = [x for x in lista_var_ok if x.startswith('v')]

var_categ_fim = [x for x in lista_var_ok if x.startswith('CT_')]
df_treino_ok = pd.concat([df_treino[['ID','target']],df_treino[var_num_fim]], axis = 1)

for var in var_categ_fim:

    dummies = pd.get_dummies(df_treino[var], prefix=var, drop_first=True)   

    df_treino_ok = pd.concat([df_treino_ok, dummies], axis=1)
X = df_treino_ok.iloc[:,2:]

Y = df_treino_ok['target']
modeloXGB = XGBClassifier(learning_rate = 0.05,

                          n_estimators = 500,

                          max_depth = 4,

                          min_child_weight = 1,

                          gamma = 0,

                          subsample = 0.7,

                          colsample_bytree = 0.5,

                          objective = 'binary:logistic',

                          n_jobs = -1,

                          nthread = -1,

                          #scale_pos_weight = 1,

                          seed = 42)
import time

start_time = time.time() 

model = modeloXGB.fit(X,Y)

print("--- %s seconds ---" % (time.time() - start_time))
prob_1 = [x[1] for x in model.predict_proba(X)]
log_loss(Y,prob_1 , eps=1e-15)
df_teste = pd.read_csv('/kaggle/input/competicao-dsa-machine-learning-dec-2019/dataset_teste.csv')

df_teste.head()
for var in var_num:

    df_teste[var] = df_teste[var].fillna(df_dsa[var].median())
df_teste.loc[df_teste['v24'] == 'A', 'CT_v24'] = 1.0

df_teste.loc[df_teste['v24'] == 'B', 'CT_v24'] = 2.0

df_teste.loc[df_teste['v24'] == 'C', 'CT_v24'] = 3.0

df_teste.loc[df_teste['v24'] == 'D', 'CT_v24'] = 1.0

df_teste.loc[df_teste['v24'] == 'E', 'CT_v24'] = 2.0



df_teste.loc[df_teste['v30'] == -99, 'CT_v30'] = 2.0

df_teste.loc[df_teste['v30'] == 'A', 'CT_v30'] = 3.0

df_teste.loc[df_teste['v30'] == 'B', 'CT_v30'] = 3.0

df_teste.loc[df_teste['v30'] == 'C', 'CT_v30'] = 2.0

df_teste.loc[df_teste['v30'] == 'D', 'CT_v30'] = 1.0

df_teste.loc[df_teste['v30'] == 'E', 'CT_v30'] = 1.0

df_teste.loc[df_teste['v30'] == 'F', 'CT_v30'] = 3.0

df_teste.loc[df_teste['v30'] == 'G', 'CT_v30'] = 1.0



df_teste.loc[df_teste['v31'] == -99, 'CT_v31'] = 2.0

df_teste.loc[df_teste['v31'] == 'A', 'CT_v31'] = 1.0

df_teste.loc[df_teste['v31'] == 'B', 'CT_v31'] = 2.0

df_teste.loc[df_teste['v31'] == 'C', 'CT_v31'] = 2.0



df_teste.loc[df_teste['v47'] == 'A', 'CT_v47'] = 2.0

df_teste.loc[df_teste['v47'] == 'B', 'CT_v47'] = 2.0

df_teste.loc[df_teste['v47'] == 'C', 'CT_v47'] = 2.0

df_teste.loc[df_teste['v47'] == 'D', 'CT_v47'] = 2.0

df_teste.loc[df_teste['v47'] == 'E', 'CT_v47'] = 1.0

df_teste.loc[df_teste['v47'] == 'F', 'CT_v47'] = 1.0

df_teste.loc[df_teste['v47'] == 'G', 'CT_v47'] = 1.0

df_teste.loc[df_teste['v47'] == 'H', 'CT_v47'] = 1.0

df_teste.loc[df_teste['v47'] == 'I', 'CT_v47'] = 1.0

df_teste.loc[df_teste['v47'] == 'J', 'CT_v47'] = 1.0



df_teste['CT_v56'] = 3.0

df_teste.loc[df_teste['v56'] == -99 ,'CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'CN','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'DF','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'DO','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'DY','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'G','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'P','CT_v56'] = 1.0

df_teste.loc[df_teste['v56'] == 'BL','CT_v56'] = 2.0

df_teste.loc[df_teste['v56'] == 'BZ','CT_v56'] = 2.0

df_teste.loc[df_teste['v56'] == 'DX','CT_v56'] = 2.0

df_teste.loc[df_teste['v56'] == 'AG','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'AW','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'BJ','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'BK','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'DA','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'DH','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'DI','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'N','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'R','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'U','CT_v56'] = 4.0

df_teste.loc[df_teste['v56'] == 'AL','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'BM','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'BQ','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'CS','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'CY','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'DR','CT_v56'] = 5.0

df_teste.loc[df_teste['v56'] == 'V','CT_v56'] = 5.0



df_teste['CT_v66'] = df_teste['v66']



df_teste['CT_v71'] = 2.0

df_teste.loc[(df_teste['v71'] == 'C') , 'CT_v71'] = 1.0



df_teste.loc[(df_teste['v79'] == 'A') | 

              (df_teste['v79'] == 'B') | 

              (df_teste['v79'] == 'F') |

              (df_teste['v79'] == 'N') |

              (df_teste['v79'] == 'O') |

              (df_teste['v79'] == 'R') |

              (df_teste['v79'] == 'J') , 'CT_v79'] = 3.0

df_teste.loc[(df_teste['v79'] == 'D') | 

              (df_teste['v79'] == 'M') |

              (df_teste['v79'] == 'Q') |

              (df_teste['v79'] == 'C') |

              (df_teste['v79'] == 'H') |

              (df_teste['v79'] == 'K') |

              (df_teste['v79'] == 'G') , 'CT_v79'] = 1.0

df_teste.loc[(df_teste['v79'] == 'E') | 

              (df_teste['v79'] == 'I') |

              (df_teste['v79'] == 'P') , 'CT_v79'] = 2.0



df_teste['CT_v110'] = 2.0

df_teste.loc[(df_teste['v110'] == 'A'), 'CT_v110'] = 1.0



df_teste['CT_v113'] = 1

df_teste.loc[(df_teste['v113'] == -99) | 

              (df_teste['v113'] == 'AD') |

              (df_teste['v113'] == 'C') |

              (df_teste['v113'] == 'E') |

              (df_teste['v113'] == 'F') |

              (df_teste['v113'] == 'H') |

              (df_teste['v113'] == 'M') |

              (df_teste['v113'] == 'N') |

              (df_teste['v113'] == 'O') |

              (df_teste['v113'] == 'T') , 'CT_v113'] = 2
df_teste_ok = pd.concat([df_teste[['ID']],df_teste[var_num_fim]], axis = 1)

for var in var_categ_fim:

    dummies = pd.get_dummies(df_teste[var], prefix=var, drop_first=True)   

    df_teste_ok = pd.concat([df_teste_ok, dummies], axis=1)
prob_1 = [x[1] for x in model.predict_proba(X)]
df_teste_ok['PredictedProb'] = [x[1] for x in model.predict_proba(df_teste_ok.iloc[:,1:])]
df_teste_ok[['ID','PredictedProb']].to_csv('18_submission.csv', index = False)