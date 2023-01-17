import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np

sns.set(style = 'whitegrid')
import os

for dirname, _, filenames in os.walk('../input/crimes-estado-rio-de-janeiro-2009-a-2019/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
parte1 = pd.read_csv('../input/crimes-estado-rio-de-janeiro-2009-a-2019/parte1.csv', 

                     sep = ';', 

                     encoding = 'iso-8859-1')



parte1 = parte1.loc[parte1.titulo_do == 'Morte por intervenção de agente do Estado']



parte1.drop(['risp', 'aisp'], axis = 1, inplace = True)





parte2 = pd.read_csv('../input/crimes-estado-rio-de-janeiro-2009-a-2019/parte2.csv', 

                     sep = ';', 

                     encoding = 'iso-8859-1')



parte2 = parte2.loc[parte2.titulo_do == 'Morte por intervenção de agente do Estado']





parte3 = pd.read_csv('../input/crimes-estado-rio-de-janeiro-2009-a-2019/parte3.csv', 

                     sep = ';', 

                     encoding = 'iso-8859-1')



parte3 = parte3.loc[parte3.titulo_do == 'Morte por intervenção de agente do Estado']





parte4 = pd.read_csv('../input/crimes-estado-rio-de-janeiro-2009-a-2019/parte4.csv', 

                     sep = ';', 

                     encoding = 'iso-8859-1')



parte4 = parte4.loc[parte4.titulo_do == 'Morte por intervenção de agente do Estado']



mortes_estado = pd.concat([parte1, parte2, parte3, parte4])

mortes_estado
serie_cisp_mortes_estado = pd.value_counts(mortes_estado['cisp'].values, sort = True)

cisp_mortes_estado = pd.DataFrame({'cisp' : serie_cisp_mortes_estado.index, 'mortos' : serie_cisp_mortes_estado.values})

pd.set_option('display.max_rows', None)

cisp_mortes_estado.set_index('cisp')
serie_idade_mortes_estado = pd.value_counts(mortes_estado['idade'].values, sort = True)

idade_mortes_estado = pd.DataFrame({'idade' : serie_idade_mortes_estado.index, 'mortos' : serie_idade_mortes_estado.values})

pd.set_option('display.max_rows', None)

idade_mortes_estado.set_index('idade')
serie_cor_mortes_estado = pd.value_counts(mortes_estado['cor'].values, sort = True)

cor_mortes_estado = pd.DataFrame({'cor' : serie_cor_mortes_estado.index, 'mortos' : serie_cor_mortes_estado.values})

pd.set_option('display.max_rows', None)

cor_mortes_estado.set_index('cor')
mortes_estado = mortes_estado[mortes_estado.cor != 'ignorado']

mortes_estado = mortes_estado[mortes_estado.cor != 'sem informação']

mortes_estado = mortes_estado[mortes_estado.cor != 'albino']

mortes_estado = mortes_estado[mortes_estado.cor != 'amarela']

mortes_estado = mortes_estado[mortes_estado.cor != 'índio']

mortes_estado = mortes_estado[mortes_estado.hora_fato != '99']

mortes_estado = mortes_estado[mortes_estado.idade != 118.0]

mortes_estado['cor'].replace({'negra': 'preta'}, inplace = True)

mortes_estado.reset_index(inplace = True)

mortes_estado.drop(['index', 'controle', 'dp'], axis = 1, inplace = True)

faixa_etaria = pd.cut(mortes_estado.idade, 

                   bins = [0, 17, 99], labels = ['menor de idade', 'maior de idade'])

mortes_estado.insert(21,'faixa etária', faixa_etaria)



mortes_estado['hora_fato'] = pd.to_datetime(mortes_estado['hora_fato'], format = '%H:%M')

mortes_estado = mortes_estado.assign(horário = pd.cut(mortes_estado.hora_fato.dt.hour,

                            [0, 6, 12, 18, 23],

                            labels = ['madrugada', 'manhã', 'tarde', 'noite'],

                            include_lowest = True))





reg = {'Zona Norte': ['039a. Pavuna', '021a. Bonsucesso', '038a. Braz de Pina', '025a. Engenho Novo', '040a. Honório Gurgel', '022a. Penha', '027a. Vicente de Carvalho', '029a. Madureira', '044a. Inhaúma', '031a. Ricardo Albuquerque', '037a. Ilha do Governador', '024a. Piedade', '030a. Marechal Hermes', '019a. Tijuca', '026a. Todos os Santos', '045a. Alemão', '020a. Grajaú', '028a. Campinho', '023a. Meier', '018a. Praça da Bandeira'], 

        'Zona Oeste': ['034a. Bangu', '036a. Santa Cruz', '032a. Taquara', '033a. Realengo', '035a. Campo Grande', '041a. Tanque', '016a. Barra da Tijuca', '042a. Recreio', '043a. Pedra de Guaratiba'], 

        'Zona Sul e Centro': ['006a. Cidade Nova', '017a. São Cristóvão', '007a. Santa Tereza', '004a. Praça da República', '011a. Rocinha', '015a. Gávea', '012a. Copacabana', '009a. Catete', '005a. Mem de Sá', '013a. Ipanema', '010a. Botafogo', '014a. Leblon'], 

        'Região Metropolitana (exceto capital)': ['054a. Belford Roxo', '059a. Duque de Caxias', '078a. Fonseca', '074a. Alcantara', '056a. Comendador Soares', '064a. Vilar dos Teles', '062a. Imbariê', '060a. Campos Elíseos', '073a. Neves', '072a. São Gonçalo', '050a. Itaguaí', '075a. Rio do Ouro', '071a. Itaboraí', '053a. Mesquita', '063a. Japerí', '077a. Icaraí', '079a. Jurujuba', '058a. Posse', '055a. Queimados', '076a. Niterói - Centro', '065a. Magé', '052a. Nova Iguaçu', '081a. Itaipú', '066a. Piabetá', '082a. Maricá', '057a. Nilópolis', '105a. Petrópolis', '048a. Seropédica', '051a. Paracambí', '067a. Guapimirim', '159a. Cachoeira de Macacú', '070a. Tanguá', '061a. Xerém', '119a. Rio Bonito', '106a. Itaipava'],

        'Interior': ['166a. Angra dos Reis', '123a. Macaé', '126a. Cabo Frio', '093a. Volta Redonda', '118a. Araruama', '151a. Nova Friburgo', '125a. São Pedro da Aldeia', '110a. Teresópolis', '146a. Guarus', '121a. Casimiro de Abreu', '124a. Saquarema', '090a. Barra Mansa', '134a. Campos', '128a. Rio das Ostras', '127a. Búzios', '167a. Parati', '165a. Mangaratiba', '089a. Resende', '132a. Arraial do Cabo', '099a. Itatiaia', '108a. Três Rios', '136a. Santo Antonio de Pádua', '147a. São Francisco de Itabapoana', '122a. Conceição de Macabú', '129a. Iguaba', '130a. Quissamã', '137a. Miracema', '138a. Lajes de Muriaré', '088a. Barra do Piraí', '145a. São João da Barra', '112a. Carmo', '139a. Porciúncula', '098a. Paulo de Frontin', '120a. Silva Jardim', '097a. Mendes', '143a. Itaperuna', '091a. Valença']}



dreg = {k: oldk for oldk, oldv in reg.items() for k in oldv}



mortes_estado['região'] = mortes_estado['cisp'].map(dreg)







mortes_estado
ax1 = sns.countplot(y = 'horário', color = 'c', data = mortes_estado)

ax1.set_xlabel('mortes');
ax2 = sns.countplot(y = 'faixa etária', data = mortes_estado, 

              hue = mortes_estado.cor.sort_values(), palette = 'Blues');

ax2.set_xlabel('mortes');
mortes_região_cor = mortes_estado.groupby(['região', 'cor']).agg({'cor': ['size']})

mortes_região_cor.columns = mortes_região_cor.columns.map('_'.join)

mortes_região_cor = mortes_região_cor.reset_index()

mortes_região_cor.rename(columns={'cor_size': 'mortes'}, inplace=True)

mortes_região_cor
mrcb = sns.catplot(x = 'cor', y = 'mortes', col = 'região', col_wrap = 2, data = mortes_região_cor, 

                   kind = 'bar', palette = 'Blues', height = 4, aspect = .8)
mortes_ano_cor = mortes_estado.groupby(['ano', 'cor']).agg({'cor': ['size']})

mortes_ano_cor.columns = mortes_ano_cor.columns.map('_'.join)

mortes_ano_cor = mortes_ano_cor.reset_index()

mortes_ano_cor.rename(columns={'cor_size': 'mortes'}, inplace=True)

mortes_ano_cor
ax = sns.lineplot(x = 'ano', y = 'mortes', hue = 'cor',

                  data = mortes_ano_cor, palette = 'Blues')