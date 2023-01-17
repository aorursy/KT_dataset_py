from ipywidgets import interact, interactive, fixed, interact_manual
from scipy.stats import f_oneway

import ipywidgets as widgets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import glob
import tqdm
dfs = []
anos = [2014, 2015, 2016, 2017]

for ano in anos:
    base = "../input/{}_{{}}.csv".format(ano)
    scouts = pd.read_csv(base.format('scouts'))
    partidas = pd.read_csv(base.format('partidas'))
    clubes = pd.read_csv(base.format('clubes'))
    jogadores = pd.read_csv(base.format('atletas'))
    if   ano in [2014]:
        data = scouts.merge(right=partidas, how='inner', left_on='partida_id', right_on='id', suffixes=['', '_partida'])
        data =   data.merge(right=clubes, how='inner', left_on='clube_id', right_on='id', suffixes=['', '_clube'])
        data =   data.merge(right=jogadores, how='inner', left_on='atleta_id', right_on='id', suffixes=['', '_atleta'])
    elif ano in [2015, 2016]:
        partidas = pd.melt(partidas, id_vars = ['id', 'rodada', 'placar_oficial_mandante', 'placar_oficial_visitante'], var_name='mando', value_name='clube_id')
        data = scouts.merge(right=partidas, how='inner', left_on=['rodada', 'clube_id'], right_on=['rodada', 'clube_id'])
        data =   data.merge(right=clubes, how='inner', left_on='clube_id', right_on='id', suffixes=['', '_clube'])
        data =   data.merge(right=jogadores, how='inner', left_on='atleta_id', right_on='id', suffixes=['', '_atleta'])
    elif ano in [2017]:
        data = scouts.merge(right=partidas, how='inner', left_on='rodada_id', right_on='rodada_id', suffixes=['', '_partida'])
        data =   data.merge(right=clubes, how='inner', left_on='clube_id', right_on='id', suffixes=['', '_clube'])
    data['ano'] = ano
    dfs.append(data)
df = pd.concat(dfs, ignore_index=True)
df = df.merge(pd.read_csv('../input/posicoes.csv'), how='left', left_on='posicao_id', right_on='id', suffixes=['', '_posicao'])
df = df[df['valida']!=False]

# Mando de Campo
df.loc[df['mando']=='clube_casa_id', 'mando'] = 1
df.loc[df['mando']=='clube_visitante_id', 'mando'] = 0
df.loc[df['ano']==2017, 'mando'] = df.loc[df['ano']==2017].apply(lambda x: (x['clube_casa_id']==x['clube_id'])*1, axis=1)

# Rodada
df.loc[df['ano']==2017,'rodada'] = df.loc[df['ano']==2017,'rodada_id']

# Investigar ->
df[df['placar_oficial_mandante'].isna()]['ano'].value_counts();

df.drop(['aproveitamento_mandante', 'aproveitamento_visitante', 'clube_casa_posicao', 'clube_visitante_posicao', 'nota', 'participou', 'partida_data', 'titular', 'substituido', 'tempo_jogado', 'status_id'], axis=1, inplace=True) # Dados insuficientes
df.drop(['id', 'clube_casa_id', 'clube_visitante_id', 'id_atleta', 'id_clube', 'partida_id', 'rodada_id', 'rodada_partida', 'atleta_id', 'clube_id', 'posicao_id', 'id_posicao', 'clube_id_atleta', 'posicao_id_atleta'], axis=1, inplace=True) # Já foi utilizado e não vai servir mais
df.drop(['local', 'abreviacao_posicao', 'slug', 'apelido', 'mando', 'valida', 'media_num'], axis=1, inplace=True) # Não serão utilizados

df = df.loc[df['placar_oficial_mandante'].notna(), :]

df.drop(['jogos_num'], axis=1, inplace=True) # Tratar depois
plt.figure(figsize=(20, 5))
sns.distplot(df[df['pontos_num'] != 0]['pontos_num'], bins=60)
plt.title('Distribuição das pontuações', fontsize=20)
plt.show();
scores = df[df['pontos_num'] != 0].groupby(by=['nome_posicao'], as_index=True).agg({'pontos_num': ['mean', lambda x: x.std(ddof=0)]})
scores.columns = ['mean', 'std']
scores.reindex(columns=scores.columns)
scores = scores.merge(pd.read_csv('../input/posicoes.csv'), left_index=True, right_on='id')

plt.figure(figsize=(20, 5))
sns.barplot(df['nome_posicao'], df['pontos_num'], ci='sd', capsize=0.2);
plt.title("Pontuação média vs. Posição do Jogador")
plt.legend()
plt.show();
valid_scores = df[(df['pontos_num'] != 0)&(np.isfinite(df['pontos_num']))]
posicoes = []
for posicao in df[df['nome_posicao'].notna()]['nome_posicao'].unique():
    posicoes.append(valid_scores[valid_scores['nome_posicao'] == posicao]['pontos_num'].values)
print("F test for ANOVA: {:.0f}, p-value = {:.2f}".format(f_oneway(*posicoes).statistic, f_oneway(*posicoes).pvalue))
scores_i = df[df['pontos_num'] != 0].groupby(by=['abreviacao'], as_index=True).agg({'pontos_num': ['mean', lambda x: x.std(ddof=0)]})
scores_i.columns = ['mean', 'std']
scores_i.reindex(columns=scores.columns)
scores_i = scores_i.sort_values(by='mean', ascending=False)

plt.figure(figsize=(20, 5))
plt.bar(scores_i.index, scores_i['mean'], yerr=scores_i['std'], capsize=10);
plt.show()
from scipy.ndimage.filters import uniform_filter1d
def show_historico_time(time_abreviacao):
    time = df[df['abreviacao']==time_abreviacao]
    scores = time[time['pontos_num'] != 0].groupby(by=['rodada', 'ano'], as_index=False).agg({'pontos_num': ['mean', lambda x: x.std(ddof=0)]})
    scores.columns = ['rodada', 'ano', 'mean', 'std']
    scores.reindex(columns=scores.columns);
    plt.figure(figsize=(20, 4))
    for ano in scores['ano'].unique():
        plt.errorbar(scores[scores['ano']==ano]['rodada'], uniform_filter1d(scores[scores['ano']==ano]['mean'], size=20), fmt='o', linestyle="-", label=ano);
    plt.title(u'Pontuação média ao longo das rodadas ({}), média móvel'.format(time_abreviacao))
    plt.legend()
    plt.show();

interact(show_historico_time, time_abreviacao=df['abreviacao'].unique());
plt.figure(figsize=(30, 10))
sns.heatmap(df.corr().abs())
plt.show();
df.info()
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.metrics import r2_score, make_scorer
import xgboost
rgs = xgboost.XGBRegressor()

#.sample(frac=1).reset_index(drop=True)
X = pd.get_dummies(df.drop(['nome', 'pontos_num'], axis=1), columns=['abreviacao', 'nome_posicao']).iloc[:, 21:]
y = df['pontos_num']
cross_val_score(rgs, X, y, cv=6, n_jobs=-1, scoring=make_scorer(r2_score))
rgs.fit(X, y)
r = pd.DataFrame(rgs.feature_importances_)
r['im'] = X.columns
r
