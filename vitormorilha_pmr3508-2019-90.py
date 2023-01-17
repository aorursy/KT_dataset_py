import pandas as pd

import sklearn
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=[0])

teste = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "id", "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

        skiprows=[0]

        )

teste=teste.drop(teste.columns[0],axis=1)

adult.head()

teste.head()

adult.isnull().sum(axis = 0)
'''

Aqui eu substituo, com uma funcao criada, alguns fatores qualitativos por quantitativos

'''



#pais_rpc = ['United-States', 27776, 'Cambodia', 270, 'England', 19709, 'Puerto-Rico', 10876, 'Canada', 19859,

#    'Germany', 27087, 'Outlying-US(Guam-USVI-etc)', 27776, 'India', 342, 'Japan', 39268, 'Greece', 11091,

#    'South', 2000, 'China', 473, 'Cuba', 2621, 'Iran', 1202, 'Honduras', 618, 'Philippines', 939, 'Italy', 19273,

#    'Poland', 2874, 'Jamaica', 2156, 'Vietnam', 220, 'Mexico', 5715, 'Portugal', 9978, 'Ireland', 15921,

#    'France', 23496, 'Dominican-Republic', 1891, 'Laos', 325, 'Ecuador', 2028, 'Taiwan', 13500, 'Haiti', 282,

#    'Columbia', 2218, 'Hungary', 4172, 'Guatemala', 1276, 'Nicaragua', 854, 'Scotland', 19709, 'Thailand', 2490,

#    'Yugoslavia', 4601, 'El-Salvador', 1384, 'Trinadad&Tobago', 3956, 'Peru', 1800, 'Hong', 22502,

#    'Holand-Netherlands', 24331]

pais_rpc = ['United-States', 6, 'Cambodia', 3, 'England', 5, 'Puerto-Rico', 5, 'Canada', 5,

    'Germany', 6, 'Outlying-US(Guam-USVI-etc)', 5, 'India', 3, 'Japan', 6, 'Greece', 5,

    'South', 4, 'China', 3, 'Cuba', 4, 'Iran', 4, 'Honduras', 3, 'Philippines', 3, 'Italy', 5,

    'Poland', 4, 'Jamaica', 4, 'Vietnam', 3, 'Mexico', 4.5, 'Portugal', 5, 'Ireland', 5,

    'France', 5, 'Dominican-Republic', 4, 'Laos', 3, 'Ecuador', 4, 'Taiwan', 5, 'Haiti', 3,

    'Columbia', 4, 'Hungary', 4, 'Guatemala', 4, 'Nicaragua', 3, 'Scotland', 5, 'Thailand', 4,

    'Yugoslavia', 4, 'El-Salvador', 4, 'Trinadad&Tobago', 4, 'Peru', 4, 'Hong', 5,

    'Holand-Netherlands', 5]



raca = ['White', 0, 'Asian-Pac-Islander',  1, 'Amer-Indian-Eskimo', 1, 'Other', 2, 'Black', 2]



classe = ['Private', 7, 'Self-emp-not-inc', 5, 'Self-emp-inc', 5, 'Federal-gov', 6, 'Local-gov', 4, 'State-gov', 5,

    'Without-pay', -5, 'Never-worked', -5, 'Bachelors', 2]



relacao= ['Wife', 4, 'Own-child', 5,'Husband', 3,'Not-in-family', 1,'Other-relative', 2,'Unmarried', 0, 'Bachelors', 2]



ocupacao=['Tech-support', 2,'Craft-repair', 2,'Other-service', 2,'Sales', 3,'Exec-managerial', 5,'Prof-specialty', 5,

   'Handlers-cleaners', 2,'Machine-op-inspct', 2,'Adm-clerical', 4,'Farming-fishing', 1,'Transport-moving', 2,

   'Priv-house-serv', 2,'Protective-serv', 4,'Armed-Forces', 5, 'Bachelors', 2]



status = ['Married-civ-spouse', 4,'Divorced', 1,'Never-married', 0,'Separated', 2,'Widowed', 6,

          'Married-spouse-absent', 3,'Married-AF-spouse', 5, 'Bachelors', 2]



adult=adult.replace(['Male', 'Female'],[1, 0])

adult=adult.replace(['<=50K', '>50K'],[1, 0])

teste=teste.replace(['Male', 'Female'],[1, 0])

teste=teste.replace(['<=50K', '>50K'],[1, 0])







def substitui(lista):

    l_old=[]

    l_new=[]

    for i in range(len(lista)):

        if i%2 == 0:

            l_old.append(lista[i])

        else:

            l_new.append(lista[i])

        i+=1

    return(l_old, l_new)



lista_rel, lista_num_rel = substitui(relacao)

adult = adult.replace(lista_rel, lista_num_rel)

teste = teste.replace(lista_rel, lista_num_rel)

   

lista_paises, lista_rpc=substitui(pais_rpc)

adult = adult.replace(lista_paises, lista_rpc)

teste = teste.replace(lista_paises, lista_rpc)

#occ1dec = pd.get_dummies(adult['Contry'])



lista_raca, lista_num_raca=substitui(raca)

adult = adult.replace(lista_raca, lista_num_raca)

teste = teste.replace(lista_raca, lista_num_raca)



lista_classe, lista_num_classe=substitui(classe)

adult=adult.replace(lista_classe,lista_num_classe)

teste=teste.replace(lista_classe,lista_num_classe)



lista_ocup, lista_num_ocup=substitui(ocupacao)

adult=adult.replace(lista_ocup, lista_num_ocup)

teste=teste.replace(lista_ocup, lista_num_ocup)



lista_status, lista_num_status=substitui(status)

adult=adult.replace(lista_status, lista_num_status)

teste=teste.replace(lista_status, lista_num_status)





adult.mean(axis = 0, skipna = True)

#adult



dados=adult.drop('Education', axis=1)

teste=teste.drop('Education', axis=1)



dados['Workclass'].fillna(6.433653, inplace = True)

dados['Occupation'].fillna(3.176476, inplace = True)

dados['Country'].fillna(5.848349, inplace = True)

dados.isnull().sum(axis = 0)

dados.drop_duplicates(inplace=True)

dados





teste['Workclass'].fillna(6.433653, inplace = True)

teste['Occupation'].fillna(3.176476, inplace = True)

teste['Country'].fillna(5.848349, inplace = True)

teste.isnull().sum(axis = 0)

teste.drop_duplicates(inplace=True)

teste.head()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split





#dados = dados.dropna()



treino=dados







Xtreino = treino[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation","Race","Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Ytreino= treino.Target

Xteste = teste[["Age", "Workclass", "Education-Num", "Martial Status",

        "Occupation","Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"]]

Yteste=teste.Target



from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



knn = KNeighborsClassifier(n_neighbors=38, p=1)



scores = cross_val_score(knn, Xtreino, Ytreino, cv=10)

scores



#accuracy_score(Yteste,YtestePred)
scores.mean()
knn.fit(Xtreino,Ytreino)

YtestePred = knn.predict(Xteste)

YtestePred
YtestePred.shape
predict = pd.DataFrame(teste)




predict=predict.drop('Target', axis=1)

predict=predict.drop('Age', axis=1)

predict=predict.drop("Workclass", axis=1)

predict=predict.drop("Education-Num", axis=1)

predict=predict.drop("Martial Status", axis=1)

predict=predict.drop("Occupation", axis=1)

predict=predict.drop("Race", axis=1)

predict=predict.drop("Sex", axis=1)

predict=predict.drop("Capital Gain", axis=1)

predict=predict.drop("Capital Loss", axis=1)

predict=predict.drop("Hours per week", axis=1)

predict=predict.drop("Country", axis=1)

predict=predict.drop("fnlwgt", axis=1)

predict=predict.drop("Relationship", axis=1)

predict["Target"] = YtestePred

predict=predict.replace([1, 0], ['<=50K', '>50K'])

predict.head()
predict.to_csv("prediction.csv", index=False)