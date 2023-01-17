import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

from kmodes.kmodes import KModes
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Definição dos codigos de cursos utilizados no ENADE

cursos = {

1:'ADMINISTRACAO',

2:'DIREITO',

5:'MEDICINA VETERINARIA',

6:'ODONTOLOGIA',

7:'MATEMATICA',

8:'COMUNICACAO SOCIAL',

9:'LETRAS',

12:'MEDICINA',

13:'CIENCIAS ECONOMICAS',

14:'FISICA',

15:'QUIMICA',

16:'BIOLOGIA',

17:'AGRONOMIA',

18:'PSICOLOGIA',

19:'FARMACIA',

20:'PEDAGOGIA',

21:'ARQUITETURA E URBANISMO',

22:'CIENCIAS CONTABEIS',

23:'ENFERMAGEM',

24:'HISTORIA',

26:'DESIGN',

27:'FONOAUDIOLOGIA',

28:'NUTRICAO',

29:'TURISMO',

30:'GEOGRAFIA',

32:'FILOSOFIA',

35:'EDUCACAO FISICA',

36:'FISIOTERAPIA',

38:'SERVICO SOCIAL',

39:'TEATRO',

40:'COMPUTACAO',

43:'MUSICA',

51:'ZOOTECNIA',

52:'TERAPIA OCUPACIONAL',

54:'CIENCIAS SOCIAIS',

55:'BIOMEDICINA',

57:'ENGENHARIA (GRUPO I)',

58:'ENGENHARIA (GRUPO II)',

59:'ENGENHARIA (GRUPO III)',

60:'ENGENHARIA (GRUPO IV)',

61:'ENGENHARIA (GRUPO V)',

62:'ENGENHARIA (GRUPO VI)',

63:'ENGENHARIA (GRUPO VII)',

64:'ENGENHARIA (GRUPO VIII)',

65:'ARQUIVOLOGIA',

66:'BIBLIOTECONOMIA',

67:'SECRETARIADO EXECUTIVO',

68:'NORMAL SUPERIOR',

69:'TECNOLOGIA EM RADIOLOGIA',

70:'TECNOLOGIA EM AGROINDUSTRIA',

71:'TECNOLOGIA EM ALIMENTOS',

72:'TECNOLOGIA EM ANALISE E DESENVOLVIMENTO DE SISTEMAS',

73:'TECNOLOGIA EM AUTOMACAO INDUSTRIAL',

74:'TECNOLOGIA EM CONSTRUCAO DE EDIFICIOS',

75:'TECNOLOGIA EM FABRICACAO MECANICA',

76:'TECNOLOGIA EM GESTAO DA PRODUCAO INDUSTRIAL',

77:'TECNOLOGIA EM MANUTENCAO INDUSTRIAL',

78:'TECNOLOGIA EM PROCESSOS QUIMICOS',

79:'TECNOLOGIA EM REDES DE COMPUTADORES',

80:'TECNOLOGIA EM SANEAMENTO AMBIENTAL',

81:'RELACOES INTERNACIONAIS',

82:'ESTATISTICA',

83:'TECNOLOGIA EM DESIGN DE MODA',

84:'TECNOLOGIA EM MARKETING',

85:'TECNOLOGIA EM PROCESSOS GERENCIAIS',

86:'TECNOLOGIA EM GESTAO DE RECURSOS HUMANOS',

87:'TECNOLOGIA EM GESTAO FINANCEIRA',

88:'TECNOLOGIA EM GASTRONOMIA',

89:'TECNOLOGIA EM GESTAO DE TURISMO',

90:'TECNOLOGIA EM AGRONEGÓCIOS',

91:'TECNOLOGIA EM GESTÃO HOSPITALAR',

92:'TECNOLOGIA EM GESTÃO AMBIENTAL',

93:'TECNOLOGIA EM GESTÃO COMERCIAL',

94:'TECNOLOGIA EM LOGÍSTICA',

701:'MATEMÁTICA (BACHARELADO)',

702:'MATEMÁTICA (LICENCIATURA)',

803:'JORNALISMO',

804:'PUBLICIDADE E PROPAGANDA',

901:'LETRAS (BACHARELADO)',

902:'LETRAS (LICENCIATURA)',            

903:'LETRAS-PORTUGUÊS (BACHARELADO)',

904:'LETRAS-PORTUGUÊS (LICENCIATURA)',

905:'LETRAS-PORTUGUÊS E INGLÊS (LICENCIATURA)',

906:'LETRAS-PORTUGUÊS E ESPANHOL (LICENCIATURA)',

1401:'FÍSICA (BACHARELADO)',

1402:'FÍSICA (LICENCIATURA)',

1501:'QUÍMICA (BACHARELADO)',

1502:'QUÍMICA (LICENCIATURA)',

1503:'QUÍMICA (ATRIBUIÇÕES TECNOLÓGICAS)',

1601:'CIÊNCIAS BIOLÓGICAS (BACHARELADO)',

1602:'CIÊNCIAS BIOLÓGICAS (LICENCIATURA)',

2001:'PEDAGOGIA (LICENCIATURA)',

2401:'HISTÓRIA (BACHARELADO)',

2402:'HISTÓRIA (LICENCIATURA)',

2501:'ARTES VISUAIS (LICENCIATURA)',

3001:'GEOGRAFIA (BACHARELADO)',

3002:'GEOGRAFIA (LICENCIATURA)',

3201:'FILOSOFIA (BACHARELADO)',

3202:'FILOSOFIA (LICENCIATURA)',

3501:'EDUCAÇÃO FÍSICA (BACHARELADO)',

3502:'EDUCAÇÃO FÍSICA (LICENCIATURA)',

4004:'CIÊNCIA DA COMPUTAÇÃO (BACHARELADO)',

4005:'CIÊNCIA DA COMPUTAÇÃO (LICENCIATURA)',

4006:'SISTEMAS DE INFORMAÇÃO',

4007:'COMPUTAÇÃO (ENGENHARIA DE COMPUTAÇÃO)',

4301:'MÚSICA (LICENCIATURA)',

5401:'CIÊNCIAS SOCIAIS (BACHARELADO)',

5402:'CIÊNCIAS SOCIAIS (LICENCIATURA)',

5707:'ENGENHARIA (GRUPO I) - ENGENHARIA GEOLÓGICA',

5708:'ENGENHARIA (GRUPO I) - ENGENHARIA DE AGRIMENSURA',

5709:'ENGENHARIA (GRUPO I) - ENGENHARIA CARTOGRÁFICA',

5710:'ENGENHARIA (GRUPO I) - ENGENHARIA CIVIL',

5711:'ENGENHARIA (GRUPO I) - ENGENHARIA DE RECURSOS HÍDRICOS',

5712:'ENGENHARIA (GRUPO I) - ENGENHARIA SANITÁRIA',

5806:'ENGENHARIA ELÉTRICA',

5807:'ENGENHARIA (GRUPO II) - ENGENHARIA INDUSTRIAL ELÉTRICA',

5808:'ENGENHARIA (GRUPO II) - ENGENHARIA ELETROTÉCNICA',

5809:'ENGENHARIA (GRUPO II) - ENGENHARIA DE COMPUTAÇÃO',

5811: 'ENGENHARIA (GRUPO II) - ENGENHARIA DE REDES DE COMUNICAÇÃO',

5812:'ENGENHARIA (GRUPO II) - ENGENHARIA ELETRÔNICA',

5813:'ENGENHARIA (GRUPO II) - ENGENHARIA MECATRÔNICA',

5814:'ENGENHARIA (GRUPO II) - ENGENHARIA DE CONTROLE E AUTOMAÇÃO',

5815:'ENGENHARIA (GRUPO II) - ENGENHARIA DE TELECOMUNICAÇÕES',

5901:'ENGENHARIA (GRUPO III) - ENGENHARIA INDUSTRIAL MECÂNICA',

5902:'ENGENHARIA (GRUPO III) - ENGENHARIA MECÂNICA',

5903:'ENGENHARIA (GRUPO III) - ENGENHARIA AEROESPACIAL',

5904:'ENGENHARIA (GRUPO III) - ENGENHARIA AERONÁUTICA',

5905:'ENGENHARIA (GRUPO III) - ENGENHARIA AUTOMOTIVA',

5906:'ENGENHARIA (GRUPO III) - ENGENHARIA NAVAL',

6005:'ENGENHARIA (GRUPO IV) - ENGENHARIA BIOQUÍMICA',

6006:'ENGENHARIA (GRUPO IV) - ENGENHARIA DE BIOTECNOLOGIA',

6007:'ENGENHARIA (GRUPO IV) - ENGENHARIA INDUSTRIAL QUÍMICA',

6008:'ENGENHARIA (GRUPO IV) - ENGENHARIA QUÍMICA',

6009:'ENGENHARIA (GRUPO IV) - ENGENHARIA DE ALIMENTOS',

6011:'ENGENHARIA (GRUPO IV) - ENGENHARIA TÊXTIL',

6106:'ENGENHARIA (GRUPO V) - ENGENHARIA DE MATERIAIS',

6107:'ENGENHARIA (GRUPO V) - ENGENHARIA FÍSICA',

6108:'ENGENHARIA (GRUPO V) - ENGENHARIA METALÚRGICA',

6109:'ENGENHARIA (GRUPO V) - ENGENHARIA DE MATERIAIS (MADEIRA)',

6110:'ENGENHARIA (GRUPO V) - ENGENHARIA DE MATERIAIS (PLÁSTICO)',

6208:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO',

6209:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO MECÂNICA',

6210:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO ELÉTRICA',

6211:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO QUÍMICA',

6213:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO DE MATERIAIS',

6214:'ENGENHARIA (GRUPO VI) - ENGENHARIA DE PRODUÇÃO CIVIL',

6306:'ENGENHARIA (GRUPO VII) - ENGENHARIA',

6307:'ENGENHARIA (GRUPO VII) - ENGENHARIA AMBIENTAL',

6308:'ENGENHARIA (GRUPO VII) - ENGENHARIA INDUSTRIAL',

6309:'ENGENHARIA (GRUPO VII) - ENGENHARIA DE MINAS',

6310:'ENGENHARIA (GRUPO VII) - ENGENHARIA DE PETRÓLEO',

6404:'ENGENHARIA (GRUPO VIII) - ENGENHARIA AGRÍCOLA',

6405:'ENGENHARIA (GRUPO VIII) - ENGENHARIA FLORESTAL',

6406:'ENGENHARIA (GRUPO VIII) - ENGENHARIA DE PESCA'

}
# Códigos para cada area

humanas = [1,2,8,9,13,18,20,24,29,30,32,38,52,54,65,66,67,68,81,803,804,903,904,905,906,2001,2401,2402,3001,3002,3201,3202,5401,5402]

saude   = [5,6,12,19,23,27,28,35,36]

arte    = [26,39,43,2501,4301]
def area(nr):

    if nr in humanas:

        return 'humanas'    

    elif nr in saude:

        return 'saude'

    elif nr in arte:

        return 'arte'

    else:

        return 'exatas'

    
estados = {

11:'RO', 28:'SE',

12:'AC', 29:'BA',

13:'AM', 31:'MG',

14:'RR', 32:'ES',

15:'PA', 33:'RJ',

16:'AP', 35:'SP',

17:'TO', 41:'PR',

21:'MA', 42:'SC',

22:'PI', 43:'RS',

23:'CE', 50:'MS',

24:'RN', 51:'MT',

25:'PB', 52:'GO',

26:'PE', 53:'DF',

27:'AL'

}
categoria_2004 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Particular'}

categoria_2005 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Particular'}

categoria_2006 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Particular'}

categoria_2007 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Pública Municipal Autarquia',5:'Privada',6:'Organização Social'}

categoria_2008 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Pública Municipal Autarquia',5:'Privada',6:'Organização Social'}

categoria_2009 = {1:'Federal', 2:'Estadual',3:'Municipal',4:'Pública Municipal Autarquia',5:'Privada',6:'Organização Social'}



categoria = {

93:'Pessoa Jurídica de Direito Público - Federal',

116:'Pessoa Jurídica de Direito Público - Municipal',

118:'Pessoa Jurídica de Direito Privado - Com fins lucrativos - Sociedade Civil',

121:'Pessoa Jurídica de Direito Privado - Sem fins lucrativos - Fundação',

10001:'Pessoa Jurídica de Direito Público - Estadual',

10002:'Pessoa Jurídica de Direito Público - Federal',

10003:'Pessoa Jurídica de Direito Público - Municipal',

10004:'Pessoa Jurídica de Direito Privado - Com fins lucrativos - Associação de Utilidade Pública',

10005:'Privada com fins lucrativos',

10006:'Pessoa Jurídica de Direito Privado - Com fins lucrativos - Sociedade Mercantil ou Comercial',

10007:'Pessoa Jurídica de Direito Privado - Sem fins lucrativos - Associação de Utilidade Pública',

10008:'Privada sem fins lucrativos',

10009:'Pessoa Jurídica de Direito Privado - Sem fins lucrativos - Sociedade',

10036:'Não disponível'

}
cat_2004 = {1:'Pública', 2:'Pública',3:'Pública',4:'Privada'}

cat_2005 = {1:'Pública', 2:'Pública',3:'Pública',4:'Privada'}

cat_2006 = {1:'Pública', 2:'Pública',3:'Pública',4:'Privada'}

cat_2007 = {1:'Pública', 2:'Pública',3:'Pública',4:'Pública',5:'Privada',6:'Privada'}

cat_2008 = {1:'Pública', 2:'Pública',3:'Pública',4:'Pública',5:'Privada',6:'Privada'}

cat_2009 = {1:'Pública', 2:'Pública',3:'Pública',4:'Pública',5:'Privada',6:'Privada'}



cat = {

93:'Pública',

116:'Pública',

118:'Privada',

121:'Privada',

10001:'Pública',

10002:'Pública',

10003:'Pública',

10004:'Privada',

10005:'Privada',

10006:'Privada',

10007:'Privada',

10008:'Privada',

10009:'Privada',

10036:'Não disponível'

}
renome = {'NU_ANO':'ano','CD_CATAD':'tipo','CO_GRUPO':'curso','co_uf_habil':'cd_uf','TP_SEXO':'tp_sexo','tp_semest':'semestre','IN_GRAD':'ingresso'}

renome1 = {'nu_ano':'ano','cd_catad':'tipo','co_grupo':'curso','co_uf_habil':'cd_uf','tp_sexo':'tp_sexo','tp_semest':'semestre','in_grad':'ingresso'}

renome2 = {'NU_ANO':'ano','CO_CATEGAD':'tipo','CO_GRUPO':'curso','CO_UF_CURSO':'cd_uf','TP_SEXO':'tp_sexo','TP_SEMESTRE':'semestre','TP_INSCRICAO':'ingresso','AMOSTRA':'amostra','NT_OBJ_FG':'nt_obj_fg','NT_OBJ_CE':'nt_obj_ce','NT_DIS_CE':'nt_dis_ce','NT_CE':'nt_ce','NT_GER':'nt_ger'}
# estas são as colunas a serem utilizadas na nossa pesquisa

colunas =  ['ano','tipo',  'curso','cd_uf','tp_sexo','semestre',  'ingresso',     'amostra','nt_obj_fg','nt_obj_ce','nt_dis_ce', 'nt_ce', 'nt_ger']

#colunas2 = ['NU_ANO','CO_CATEGAD','CO_GRUPO','CO_UF_CURSO','TP_SEXO','TP_SEMESTRE','TP_INSCRICAO','AMOSTRA','NT_OBJ_FG','NT_OBJ_CE','NT_DIS_CE', 'NT_CE', 'NT_GER']
df_2014 = pd.read_csv("/kaggle/input/enadezip/MICRODADOS_ENADE_2014.txt", sep=";") 

df_2014 = df_2014.rename(columns=renome2)

data_2014 = df_2014.loc[:,colunas]

data_2014.loc[:,'categoria'] = data_2014["tipo"].map(categoria)

data_2014.loc[:,'cat'] = data_2014["tipo"].map(cat)

data_2014.loc[:,'grupo'] = data_2014["curso"].map(cursos)

data_2014.loc[:,'uf'] = data_2014["cd_uf"].map(estados)

data_2014.loc[:,'area'] = data_2014['curso'].apply(area) 

data_2014.info()
df_2013 = pd.read_csv("/kaggle/input/enadezip/MICRODADOS_ENADE_2013.txt", sep=";") 

df_2013 = df_2013.rename(columns=renome2)

data_2013 = df_2013.loc[:,colunas]

data_2013.loc[:,'categoria'] = data_2013["tipo"].map(categoria)

data_2013.loc[:,'cat'] = data_2013["tipo"].map(cat)

data_2013.loc[:,'grupo'] = data_2013["curso"].map(cursos)

data_2013.loc[:,'uf'] = data_2013["cd_uf"].map(estados)

data_2013.loc[:,'area'] = data_2013['curso'].apply(area) 

data_2013.info()
df_2012 = pd.read_csv("/kaggle/input/enadezip/MICRODADOS_ENADE_2012.txt", sep=";") 

df_2012 = df_2012.rename(columns=renome2)

data_2012 = df_2012.loc[:,colunas]

data_2012.loc[:,'categoria'] = data_2012["tipo"].map(categoria)

data_2012.loc[:,'cat'] = data_2012["tipo"].map(cat)

data_2012.loc[:,'grupo'] = data_2012["curso"].map(cursos)

data_2012.loc[:,'uf'] = data_2012["cd_uf"].map(estados)

data_2012.loc[:,'area'] = data_2012['curso'].apply(area) 

data_2012.info()
df_2011 = pd.read_csv("/kaggle/input/enadezip/MICRODADOS_ENADE_2011.txt", sep=";") 

df_2011 = df_2011.rename(columns=renome2)

data_2011 = df_2011.loc[:,colunas]

data_2011.loc[:,'categoria'] = data_2011["tipo"].map(categoria)

data_2011.loc[:,'cat'] = data_2011["tipo"].map(cat)

data_2011.loc[:,'grupo'] = data_2011["curso"].map(cursos)

data_2011.loc[:,'uf'] = data_2011["cd_uf"].map(estados)

data_2011.loc[:,'area'] = data_2011['curso'].apply(area) 

data_2011.info()
df_2010 = pd.read_csv("/kaggle/input/enadezip/MICRODADOS_ENADE_2010.txt", sep=";") 

df_2010 = df_2010.rename(columns=renome2)

data_2010 = df_2010.loc[:,colunas]

data_2010.loc[:,'categoria'] = data_2010["tipo"].map(categoria)

data_2010.loc[:,'cat'] = data_2010["tipo"].map(cat)

data_2010.loc[:,'grupo'] = data_2010["curso"].map(cursos)

data_2010.loc[:,'uf'] = data_2010["cd_uf"].map(estados)

data_2010.loc[:,'area'] = data_2010['curso'].apply(area) 

data_2010.info()
df_2009 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2009.csv", sep=";", encoding='latin1') 

df_2009 = df_2009.rename(columns=renome1)

data_2009 = df_2009.loc[:,colunas]

data_2009.loc[:,'categoria'] = data_2009["tipo"].map(categoria_2009)

data_2009.loc[:,'cat'] = data_2009["tipo"].map(cat_2009)

data_2009.loc[:,'grupo'] = data_2009["curso"].map(cursos)

data_2009.loc[:,'uf'] = data_2009["cd_uf"].map(estados)

data_2009.loc[:,'area'] = data_2009['curso'].apply(area) 

data_2009.info()
df_2008 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2008.csv", sep=";") 

df_2008 = df_2008.rename(columns=renome1)

data_2008 = df_2008.loc[:,colunas]

data_2008.loc[:,'categoria'] = data_2008["tipo"].map(categoria_2008)

data_2008.loc[:,'cat'] = data_2008["tipo"].map(cat_2008)

data_2008.loc[:,'grupo'] = data_2008["curso"].map(cursos)

data_2008.loc[:,'uf'] = data_2008["cd_uf"].map(estados)

data_2008.loc[:,'area'] = data_2008['curso'].apply(area) 

data_2008.info()
df_2007 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2007.csv", sep=";", encoding='latin1') 

df_2007 = df_2007.rename(columns=renome1)

data_2007 = df_2007.loc[:,colunas]

data_2007.loc[:,'categoria'] = data_2007["tipo"].map(categoria_2007)

data_2007.loc[:,'cat'] = data_2007["tipo"].map(cat_2007)

data_2007.loc[:,'grupo'] = data_2007["curso"].map(cursos)

data_2007.loc[:,'uf'] = data_2007["cd_uf"].map(estados)

data_2007.loc[:,'area'] = data_2007['curso'].apply(area) 

data_2007.info()
df_2006 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2006.csv", sep=";") 

df_2006 = df_2006.rename(columns=renome1)

data_2006 = df_2006.loc[:,colunas]

data_2006.loc[:,'categoria'] = data_2006["tipo"].map(categoria_2006)

data_2006.loc[:,'cat'] = data_2006["tipo"].map(cat_2006)

data_2006.loc[:,'grupo'] = data_2006["curso"].map(cursos)

data_2006.loc[:,'uf'] = data_2006["cd_uf"].map(estados)

data_2006.loc[:,'area'] = data_2006['curso'].apply(area) 

data_2006.info()
df_2005 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2005.csv", sep=";",encoding='latin1') 

df_2005 = df_2005.rename(columns=renome1)

data_2005 = df_2005.loc[:,colunas]

data_2005.loc[:,'categoria'] = data_2005["tipo"].map(categoria_2005)

data_2005.loc[:,'cat'] = data_2005["tipo"].map(cat_2005)

data_2005.loc[:,'grupo'] = data_2005["curso"].map(cursos)

data_2005.loc[:,'uf'] = data_2005["cd_uf"].map(estados)

data_2005.loc[:,'area'] = data_2005['curso'].apply(area) 

data_2005.info()
#Há dois registros com ano = -2005

data_2005.loc[data_2005['ano'] == -2005, 'ano'] = 2005
df_2004 = pd.read_csv("/kaggle/input/enadezip/microdados_enade_2004.csv", sep=";",encoding='latin1')

df_2004 = df_2004.rename(columns=renome)

data_2004 = df_2004.loc[:,colunas]

data_2004.loc[:,'categoria'] = data_2004["tipo"].map(categoria_2004)

data_2004.loc[:,'cat'] = data_2004["tipo"].map(cat_2004)

data_2004.loc[:,'grupo'] = data_2004["curso"].map(cursos)

data_2004.loc[:,'uf'] = data_2004["cd_uf"].map(estados)

data_2004.loc[:,'area'] = data_2004['curso'].apply(area) 

data_2004.info()
data_2006
from scipy.stats import ttest_ind
#agrega todas as bases anuais

#calcula os p-valores para verificar que as medias são diferentes em cada ano

#imprime as estatisticas para cada ano

data_geral = pd.DataFrame

matriz_ano = []

for nr in range(2004, 2015):

    ano = str(nr)

    print('-------------------------------')

    print(ano)

    print()

    exec("privada_" + ano + " = data_" + ano + "[(data_" + ano + "['cat'] == 'Privada') & (~data_" + ano + "['nt_ger'].isnull())]['nt_ger']")

    exec("publica_" + ano + " = data_" + ano + "[(data_" + ano + "['cat'] == 'Pública') & (~data_" + ano + "['nt_ger'].isnull())]['nt_ger']")

    print('Privada')

    exec("print(privada_" + ano + ".describe())")

    print()

    print('Pública')

    exec("print(publica_" + ano + ".describe())")

    exec("print(ttest_ind(privada_" + ano + ",publica_" + ano + "))")

    exec("x,y = ttest_ind(privada_" + ano + ",publica_" + ano + ")")

    exec("matriz_ano.append([nr,x,y])")

    if nr == 2004:

        data_geral = data_2004

    else:

        exec("data_geral = data_geral.append(data_" + ano + ", sort=False)")

# p_valor_ano guarda o p_valor com   o teste para duas medias (publica e privada) para cada ano        

p_valor_ano = pd.DataFrame(matriz_ano,columns=["ano","statistic","pvalue"])

print(p_valor_ano)

p_valor_ano.to_csv('p_valor_ano.csv')
#calcula os p-valores para verificar que as medias são diferentes em cada uf

#imprime as estatisticas para cada uf

matriz_uf = []

for estado in data_geral["uf"].unique():

    print('-------------------------------')

    print(estado)

    print()

    exec("privada_" + estado + " = data_geral[(data_geral['cat'] == 'Privada') & (data_geral['uf'] == estado) & (~data_geral['nt_ger'].isnull())]['nt_ger']")

    exec("publica_" + estado + " = data_geral[(data_geral['cat'] == 'Pública') & (data_geral['uf'] == estado) & (~data_geral['nt_ger'].isnull())]['nt_ger']")

    print('Privada')

    exec("print(privada_" + estado + ".describe())")

    print()

    print('Pública')

    exec("print(publica_" + estado + ".describe())")

    exec("print(ttest_ind(privada_" + estado + ",publica_" + estado + "))")

    exec("x,y = ttest_ind(privada_" + estado + ",publica_" + estado + ")")

    exec("matriz_uf.append([estado,x,y])")

p_valor_uf = pd.DataFrame(matriz_uf,columns=["uf","statistic","pvalue"])

print(p_valor_uf)

p_valor_uf.to_csv('p_valor_uf.csv')
#Definição donuemro de casa decimais a ser apresentado nos numeros

pd.options.display.float_format = '{:.2f}'.format   #corrige o format para 2 decimais

data_geral['nt_ger'].describe()
privada_geral = data_geral[data_geral['cat'] == 'Privada']

privada_geral.describe()
#publica_geral = data_geral[(data_geral['cat'] == 'Pública') & (~data_geral['nt_ger'].isnull())]['nt_ger']

publica_geral = data_geral[data_geral['cat'] == 'Pública']

publica_geral.describe()
privada_geral[~(privada_geral['nt_ger'].isnull())]['nt_ger']
ttest_ind(privada_geral[~(privada_geral['nt_ger'].isnull())]['nt_ger'],publica_geral[~(publica_geral['nt_ger'].isnull())]['nt_ger'])
publica_geral


#Extraimos agora as medias por estado para publicas e para privadas



media_publica_por_uf = publica_geral.groupby(['uf'])['nt_ger'].mean().reset_index()

media_privada_por_uf = privada_geral.groupby(['uf'])['nt_ger'].mean().reset_index()

media_publica_por_uf = media_publica_por_uf.rename(columns={'nt_ger':'publica'})

media_privada_por_uf = media_privada_por_uf.rename(columns={'nt_ger':'privada'})
# efetuamos o join entre as medias de publicas de privadas por uf

media_por_uf = media_publica_por_uf.set_index('uf').join(media_privada_por_uf.set_index('uf')).reset_index()

media_por_uf['diferenca'] = media_por_uf['publica'] - media_por_uf['privada']
media_por_uf
# Calculamos agora as medias de publicas e privadas por ano

media_publica_anual = publica_geral.groupby(['ano'])['nt_ger'].mean().reset_index()

media_privada_anual = privada_geral.groupby(['ano'])['nt_ger'].mean().reset_index()

media_publica_anual = media_publica_anual.rename(columns={'nt_ger':'publica'})

media_privada_anual = media_privada_anual.rename(columns={'nt_ger':'privada'})
# Efetuamos o join agora para as medias de publicas e privadas por ano

media_anual = media_publica_anual.set_index('ano').join(media_privada_anual.set_index('ano')).reset_index()

media_anual['diferenca'] = media_anual['publica'] - media_anual['privada']
media_anual
%matplotlib inline

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np



pos = list(range(len(media_anual['privada']))) 

width = 0.25 

    

fig, ax = plt.subplots(figsize=(10,5))



plt.bar(pos, 

        media_anual['privada'], 

        width, 

        alpha=0.5, 

        color='#EE3224', 

        label=media_anual['ano'][0]) 



plt.bar([p + width for p in pos], 

        media_anual['publica'],

        width, 

        alpha=0.5, 

        color='#F78F1E', 

        label=media_anual['ano'][1]) 



ax.set_ylabel('Média da nota')



# Set the chart's title

ax.set_title('Médias das notas anuais')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(media_anual['ano'])



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(media_anual['privada'] + media_anual['publica'] )] )



# Adding the legend and showing the plot

plt.legend(['Privada', 'Pública'], loc='upper left')

plt.grid()

plt.show()
pos = list(range(len(media_por_uf['privada']))) 

width = 0.25 

    

fig, ax = plt.subplots(figsize=(10,5))



plt.bar(pos, 

        media_por_uf['privada'], 

        width, 

        alpha=0.5, 

        color='#EE3224', 

        label=media_por_uf['uf'][0]) 



plt.bar([p + width for p in pos], 

        media_por_uf['publica'],

        width, 

        alpha=0.5, 

        color='#F78F1E', 

        label=media_por_uf['uf'][1]) 



ax.set_ylabel('Média da nota')



# Set the chart's title

ax.set_title('Médias das notas nos estados')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(media_por_uf['uf'])



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(media_por_uf['privada'] + media_por_uf['publica'] )] )



# Adding the legend and showing the plot

plt.legend(['Privada', 'Pública'], loc='upper left')

plt.grid()

plt.show()
#Isolamos primeio os dados de Tocantins

#Depois, para este estado, separamos as medias de publicas e privadas por ano

data_geral_to =  data_geral[data_geral['uf'] == 'TO']

publica_anual_to = data_geral_to[data_geral_to['cat'] == 'Pública'].groupby(['ano'])['nt_ger'].mean().reset_index()

privada_anual_to = data_geral_to[data_geral_to['cat'] == 'Privada'].groupby(['ano'])['nt_ger'].mean().reset_index()

publica_anual_to = publica_anual_to.rename(columns={'nt_ger':'publica'})

privada_anual_to = privada_anual_to.rename(columns={'nt_ger':'privada'})
media_anual_to = publica_anual_to.set_index('ano').join(privada_anual_to.set_index('ano')).reset_index()

media_anual_to['diferenca'] = media_anual_to['publica'] - media_anual_to['privada']
pos = list(range(len(media_anual_to['privada']))) 

width = 0.25 

    

fig, ax = plt.subplots(figsize=(10,5))



plt.bar(pos, 

        media_anual_to['privada'], 

        width, 

        alpha=0.5, 

        color='#EE3224', 

        label=media_anual_to['ano'][0]) 



plt.bar([p + width for p in pos], 

        media_anual_to['publica'],

        width, 

        alpha=0.5, 

        color='#F78F1E', 

        label=media_anual_to['ano'][1]) 



ax.set_ylabel('Média da nota no Tocantins')



# Set the chart's title

ax.set_title('Médias das notas anuais em TO')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(media_anual_to['ano'])



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(media_anual_to['privada'] + media_anual_to['publica'] )] )



# Adding the legend and showing the plot

plt.legend(['Privada', 'Pública'], loc='upper left')

plt.grid()

plt.show()
data_geral_2011 =  data_geral[data_geral['ano'] == 2011]

publica_anual_2011 = data_geral_2011[data_geral_2011['cat'] == 'Pública'].groupby(['uf'])['nt_ger'].mean().reset_index()

privada_anual_2011 = data_geral_2011[data_geral_2011['cat'] == 'Privada'].groupby(['uf'])['nt_ger'].mean().reset_index()

publica_anual_2011 = publica_anual_2011.rename(columns={'nt_ger':'publica'})

privada_anual_2011 = privada_anual_2011.rename(columns={'nt_ger':'privada'})
data_geral_2011
media_anual_2011 = publica_anual_2011.set_index('uf').join(privada_anual_2011.set_index('uf')).reset_index()

media_anual_2011['diferenca'] = media_anual_2011['publica'] - media_anual_2011['privada']
pos = list(range(len(media_anual_2011['privada']))) 

width = 0.25 

    

fig, ax = plt.subplots(figsize=(10,5))



plt.bar(pos, 

        media_anual_2011['privada'], 

        width, 

        alpha=0.5, 

        color='#EE3224', 

        label=media_anual_2011['uf'][0]) 



plt.bar([p + width for p in pos], 

        media_anual_2011['publica'],

        width, 

        alpha=0.5, 

        color='#F78F1E', 

        label=media_anual_2011['uf'][1]) 



ax.set_ylabel('Média das notas')



# Set the chart's title

ax.set_title('Médias das notas nos estados em 2011')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(media_anual_2011['uf'])



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(media_anual_2011['privada'] + media_anual_2011['publica'] )] )



# Adding the legend and showing the plot

plt.legend(['Privada', 'Pública'], loc='upper left')

plt.grid()

plt.show()
data_2011_to =  data_geral[(data_geral['ano'] == 2011) & (data_geral['uf'] == 'TO')]
publica_2011_to = data_2011_to[(data_2011_to['cat'] == 'Pública') & (~data_2011_to['nt_ger'].isnull())]['nt_ger']

publica_2011_to.describe()
privada_2011_to = data_2011_to[(data_2011_to['cat'] == 'Privada') & (~data_2011_to['nt_ger'].isnull())]['nt_ger']

privada_2011_to.describe()
ttest_ind(privada_2011_to,publica_2011_to)
data_2011_to
publica_media_2011_to = data_2011_to[data_2011_to['cat'] == 'Pública'].groupby(['area'])['nt_ger'].mean().reset_index()

privada_media_2011_to = data_2011_to[data_2011_to['cat'] == 'Privada'].groupby(['area'])['nt_ger'].mean().reset_index()

publica_media_2011_to = publica_media_2011_to.rename(columns={'nt_ger':'publica'})

privada_media_2011_to = privada_media_2011_to.rename(columns={'nt_ger':'privada'})
media_2011_to = publica_media_2011_to.set_index('area').join(privada_media_2011_to.set_index('area')).reset_index()

media_2011_to['diferenca'] = media_2011_to['publica'] - media_2011_to['privada']
media_2011_to

#media_2011_to.to_csv('media_2011_to.csv')
#data_2011_to.to_csv('data_2011_to.csv')
data_2011_sp =  data_geral[(data_geral['ano'] == 2011) & (data_geral['uf'] == 'SP')]
publica_2011_sp = data_2011_sp[(data_2011_sp['cat'] == 'Pública') & (~data_2011_sp['nt_ger'].isnull())]['nt_ger']

publica_2011_sp.describe()
privada_2011_sp = data_2011_sp[(data_2011_sp['cat'] == 'Privada') & (~data_2011_sp['nt_ger'].isnull())]['nt_ger']

privada_2011_sp.describe()
ttest_ind(privada_2011_sp,publica_2011_sp)
publica_media_2011_sp = data_2011_sp[data_2011_sp['cat'] == 'Pública'].groupby(['area'])['nt_ger'].mean().reset_index()

privada_media_2011_sp = data_2011_sp[data_2011_sp['cat'] == 'Privada'].groupby(['area'])['nt_ger'].mean().reset_index()

publica_media_2011_sp = publica_media_2011_sp.rename(columns={'nt_ger':'publica'})

privada_media_2011_sp = privada_media_2011_sp.rename(columns={'nt_ger':'privada'})
media_2011_sp = publica_media_2011_sp.set_index('area').join(privada_media_2011_sp.set_index('area')).reset_index()

media_2011_sp['diferenca'] = media_2011_sp['publica'] - media_2011_sp['privada']
media_2011_sp
#Verificando a correlação

corr = data_geral.corr(method='spearman')
corr
#corr.to_csv('corr_data_geral.csv')
#corr = data_geral.corr(method='spearman')

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(9, 8))



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap="YlGnBu", vmax=.99, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
data_geral.loc[:,'sexo'] = data_geral["tp_sexo"].map({'F':0,'M':1,2:0,1:1,0.0:2,'N':2})
#Atualiza a oluna com valor null para 2

data_geral.loc[data_geral['sexo'].isnull(),'sexo'] = 2
data_geral['sexo'].value_counts()
data_geral.info()
data_geral.loc[data_geral['semestre'].isnull(),'semestre'] = 0
data_geral.isnull().sum()
#Insere columa numerica que indica se é publica ou privada

data_geral.loc[:,'catn'] = data_geral["cat"].map({'Pública':0,'Privada':1})
#Insere columa numerica que indica a area

data_geral.loc[:,'arean'] = data_geral["area"].map({'humanas':0,'exatas':1,'saude':2,'arte':3 })
#Retira os registros sem nota geral

data_geral_notas = data_geral[~data_geral['nt_ger'].isnull()]
#Retira os registros sem informação de categoria (publica/privada)

data_geral_notas = data_geral_notas[data_geral_notas['cat']!= 'Não disponível']
#incluindo categorias baseadas em faixa de valores para as notas.

data_geral_notas['nota'] = round(data_geral_notas['nt_ger']/10)
data_geral_notas
#data_geral_notas.to_csv('data_geral_notas.csv')
#Verificando as classes na coluna area

data_geral_notas['area'].value_counts()
#verificando o numero de privadas e publicas

data_geral_notas['cat'].value_counts()


#Conferindo as colunas com valores nulos novamente

data_geral_notas.isnull().sum()
#Definição das colunas a serem usadas na analise de clusters

colunas3 =  ['ano','cd_uf','sexo','semestre','ingresso', 'amostra','nota','catn','arean']
#Filtrando as colunas do dataset para análise de cluesters

#data_cluster = data_geral_notas.loc[:,colunas3]

data_cluster = pd.read_csv("/kaggle/input/data-cluster-trabalho-software-iii/data_cluster.csv") 

#Verificando os registros nulos

data_cluster.isnull().sum()
#Teste do custo para varias quantidades de clusters

# Impressão do grafico Elbow

plt.figure(figsize=(10,5))

#cost = [21154847.0, 18742677.0, 17592839.0, 17074922.0]

cost =  [17050937.0, 14715579.0, 13479271.0, 12889944.0]





K = range(1,5)



#*******   Devido ao alto custo de processamento não vamos executar repetidas vezes a rotina abaixo *******

#cost = []

#for num_clusters in list(K):

#    kmode = KModes(n_clusters=num_clusters, init = "Cao", n_init = 1, verbose=1)

#    kmode.fit_predict(data_cluster)

#    cost.append(kmode.cost_)   

# Obtemos cost = [21154847.0, 18742677.0, 17592839.0, 17074922.0]

    

plt.plot(K, cost, 'bx-')

plt.xlabel('k clusters')

plt.ylabel('Custo')

plt.title('Método Elbow para encontrar o nr ideal de clusters')

plt.show()
#Rotina para análise de clusters - Rotina Kmode para variaveis categoricas



#****   Rotina comentada devido ao tempo excessivo de processamento   **************



#km = KModes(n_clusters=2, init = "Cao", n_init = 1, verbose=1)

#cluster_labels = km.fit_predict(data_cluster)

#data_cluster['Cluster'] = cluster_labels   # Adição de uma coluna que informa o nuemro do cluster
#Visualização da tabela com a coluna Cluster

data_cluster
#Plotagem dos graficos com a distribuição de elementos em cada cluster

for col in colunas3:

    plt.subplots(figsize = (15,5))

    sns.countplot(x='Cluster',hue=col, data = data_cluster)

    plt.show()
#Separando as colunas para o treinamento - Variaveis independentes

data_copia = data_geral_notas.loc[:,['ano','cd_uf','sexo','semestre','ingresso', 'amostra','catn','arean','nt_ger']]
#import pandas as pd

#import numpy as np

import itertools

from itertools import chain, combinations

import statsmodels.formula.api as smf

import scipy.stats as scipystats

import statsmodels.api as sm

import statsmodels.stats.stattools as stools

import statsmodels.stats as stats

from statsmodels.graphics.regressionplots import *

#import matplotlib.pyplot as plt

#import seaborn as sns

import copy

#from sklearn.cross_validation import train_test_split

import math

import time
lm = smf.ols('nt_ger ~ C(cd_uf) + C(sexo) + C(semestre) + C(ingresso) + C(amostra) + C(catn) + C(arean) + ano', data = data_copia).fit()

#lm.summary()