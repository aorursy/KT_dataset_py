import numpy as np

import operator as op

import matplotlib.pyplot as plt



from pprint import *

from random import *



#read the csv data file...

f = open("../input/Iris.csv", 'r')

lines = [v.rstrip().split(',') for v in f.readlines()]

f.close()



print(lines[0])



#extract the setosa

list_iris_setosa = []

for v in lines:

    if v[-1] == "Iris-setosa":

        list_iris_setosa.append([float(w) for w in v[:-1]])

shuffle(list_iris_setosa)



#extract the versicolor

list_iris_versicolor = []

for v in lines:

    if v[-1] == "Iris-versicolor":

        list_iris_versicolor.append([float(w) for w in v[:-1]])

shuffle(list_iris_versicolor)



#extract the virginica

list_iris_virginica = []

for v in lines:

    if v[-1] == "Iris-virginica":

        list_iris_virginica.append([float(w) for w in v[:-1]])

shuffle(list_iris_virginica)



#get all iris

all_iris = list_iris_setosa + list_iris_versicolor + list_iris_virginica
class Perceptron: #an old perceptron code I made some time ago... http://orissermaroix.url.ph/code/perceptron.py

    """Classe perceptron, cette classe permet de :

     - creer un perceptron simple

     - l'entrainer

     - l'evaluer

     - l'interroger"""



    def __init__(self, size, alpha=None):

        """On cree un perceptron agissant sur des vecteurs 

        d'une taille 'size' donnee en parametre.

        Le parametre optionnel 'alpha' peut etre utile dans le cas ou

        ce qui doit etre classifie n'est pas lineairement separable... On

        reduit le taux d'aprentissage au fur et a mesure afin de converger

        ver la meilleur solution possible."""

        #on initialise les poids a 0la longeurs de poids est egale a:

        #la longeur des vecteurs a traiter + 1 pour prendre en compte

        #le biais qui sera la premiere caracteristique

        self.__w = [0] * (size + 1)

        self.__nb_iter = 0

        self.__alpha = alpha

        self.reset_stats()

    

    def size(self):

        """Retourne la taille des vecteurs traites par notre 

        perceptron, elle est egale a la taille donne en parametre

        lors de la creation de notre perceptron"""

        return len(self.__w) - 1



    def nb_iter(self):

        """Retourne ne nombre d'iterations que notre perceptron

        a du subir pour s'entrainer et etre tel qu'il est actuellement"""

        return self.__nb_iter



    def reset_stats(self):

        """Met a 0 toutes les conteurs de vrais/faux positifs/negatifs"""

        self.__tp = 0

        self.__fp = 0

        self.__tn = 0

        self.__fn = 0



    def get_true_positive(self):

        """Renvoie le nombre de vrais positifs calcules"""

        return self.__tp



    def get_false_positive(self):

        """Renvoie le nombre de faux positis calcules"""

        return self.__fp



    def get_true_negative(self):

        """Renvoie le nombde de vrais negatifs calcules"""

        return self.__tn



    def get_false_negative(self):

        """Retourne le nombde de faux negatifs calcules"""

        return self.__fn



    def get_nb_calculs(self):

        """Retourne la somme de vrais/faux positifs/negatifs calcules"""

        return self.__tp + self.__fp + self.__tn + self.__fn



    def predict(self, values_vector):

        """On prend une liste de caracteristiques

        que l'on soummet a notre perceptron afin d'obtenir

        une prediction de sa part, la longueur du vecteur

        des valeurs est egale a la taille d'initialisation

        du perceptron"""

        return sum(map(

            op.mul, 

            [1] + values_vector,

            self.__w

        )) >= 0



    def train(self, values_vector, output):

        """On prend une liste de caracteristiques et une valeur a predire

        que l'on soummet a notre perceptron afin qu'il s'entraine.

        La longeur du vecteur des valuers est egale a la taille 

        d'initialisation du perceptron tandis que 'output' est un boolean"""

        p_output = self.predict(values_vector)

        if p_output == output:

            return

        self.__nb_iter += 1

        self.__w = list(map(op.sub if p_output else op.add,

                       self.__w,

                       list(map(op.mul,

                           [self.__alpha / (1 + self.__nb_iter)] * len(self.__w),

                           [1] + values_vector))

                       if self.__alpha else [1] + values_vector

                       ))



    def fit(self, positives_lst, negatives_lst, train_with_nb_samples=500):

        """On prend la liste des exemples positifs et la liste des exemples

        negatifs et on entraine train_with_nb_samples nombre de fois notre

        perceptron avec un exemple de chaque liste choisi aleatoirement."""

        for i in range(train_with_nb_samples):

            #pick one positive & one negative sample

            positive_sample = choice(positives_lst)

            negative_sample = choice(negatives_lst)

            #train with those samples

            self.train(positive_sample, 1)

            self.train(negative_sample, 0)

        

    

    def evaluate(self, values_vector, output, train=False):

        """On prend une liste de caracteristiques et une valeur a predire

        que l'on soummet a notre perceptron afin d'evaluer ses performances.

        La longeur du vecteur des valeurs est egale a la taille d'initialisation

        du perceptron tandis que 'output' est un boolean.

        Si l'on choisi que mettre le parametre train a True, notre perceptron

        se servira de l'exemple passe en parametre pour s'entraine apres qu'il

        ait ete evalue (pour ne pas fausser nos resultats.

        On revoie un boolean pour savoir si notre perceptron a bien predit la

        bonne classe et on update son nombre de faux positifs et

        de faux negatifs"""

        p_output = self.predict(values_vector)

        if p_output == output:

            if p_output:

                self.__tp += 1

            else:

                self.__tn += 1

        else:

            if p_output:

                self.__fp += 1

            else:

                self.__fn += 1

        if train:

            self.train(values_vector, output)

        return p_output

test_perceptron = Perceptron(3)



test_perceptron.fit(

    positives_lst=[

        [0, 1, 1],

        [1, 0, 1],

        [1, 1, 1]

    ],

    negatives_lst=[[0, 0, 1]]

)



print(list(map(

    op.eq, #test if weel predicted....

    map(test_perceptron.predict, [[0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 1]]), #y hat

    [1, 1, 1, 0] #y

)))

class MultiClassPerceptron:

    def __init__(self, size, nb_classes):

        self.__classes_percetron = [Perceptron(size) for i in range(nb_classes)]

        

    def train(self, data_lst):

        """

        This method train each perceptron with the same datas.

        

        :param data_lst: one training case of each class alway in the same order.

        

        :type data_lst: list(list())

        """

        for pi, p in enumerate(self.__classes_percetron):

            for tci, tc in enumerate(data_lst):

                if tci == pi:

                    #to get the same amout of True traning than False

                    for i in range(len(self.__classes_percetron) - 1): 

                        p.train(tc, True)

                else:

                    p.train(tc, False)

                

     

    def fit(self, lst_of_tc_gouped_by_class, nb_iter=500):

        """

        This method will fit the multiclass perceptron using the given dataset.

        

        :param lst_of_tc_gouped_by_class: a list of this form :

        [

            [training case 1 class 1, training case 2 class 1, ..., training case m class 1],

            [training case 1 class 2, training case 2 class 2, ..., training case m class 2],

            ...

            [training case 1 class 1, training case 2 class n, ..., training case m class n],

        ]

        

        :type lst_of_tc_gouped_by_class: list(list(list))

        """

        X = lst_of_tc_gouped_by_class #renaming, shorted to write... :p

        for i in range(nb_iter):

            samples = [choice(X[j]) for j in range(len(X))]

            self.train(samples)

    

    def predict(self, values_vector):

        """Will return a list of boolean where each element of index i is 

        True if the perceptron of class i accept this element False else.

        

        :param values_vector: the vector to classify

        

        :type values_vector: list

        

        :return: a list of boolean with the predicted appartencane to each class

        :rtype: list(bool)

        """

        return [p.predict(values_vector) for p in self.__classes_percetron]

        
training_list_iris_setosa = list_iris_setosa[:int(len(list_iris_setosa) * .6)]

testing_list_iris_setosa  = list_iris_setosa[int(len(list_iris_setosa) * .6):int(len(list_iris_setosa) * .8)]

cross_validation_list_iris_setosa = list_iris_setosa[int(len(list_iris_setosa) * .8):]



training_list_iris_versicolor = list_iris_versicolor[:int(len(list_iris_versicolor) * .8)]

testing_list_iris_versicolor  = list_iris_versicolor[int(len(list_iris_versicolor) * .6):int(len(list_iris_versicolor) * .8)]

cross_validation_list_iris_versicolor = list_iris_versicolor[int(len(list_iris_versicolor) * .8):]



training_list_iris_virginica = list_iris_virginica[:int(len(list_iris_virginica) * .8)]

testing_list_iris_virginica  = list_iris_virginica[int(len(list_iris_virginica) * .6):int(len(list_iris_virginica) * .8)]

cross_validation_list_iris_virginica = list_iris_virginica[int(len(list_iris_virginica) * .8):]
mcp_iris_classifier = MultiClassPerceptron(5, 3)

mcp_iris_classifier.fit(

    [

        training_list_iris_setosa,

        training_list_iris_versicolor,

        training_list_iris_virginica

    ],

    5000

)
print("\nclass 1:")

print(

    "nb training case used : ",

    mcp_iris_classifier._MultiClassPerceptron__classes_percetron[0].nb_iter()

)

cross_valisation_mcp_res_setosa = list(map(mcp_iris_classifier.predict, cross_validation_list_iris_setosa))

pprint(cross_valisation_mcp_res_setosa)



print("\nclass 2:")

print(

    "nb training case used : ",

    mcp_iris_classifier._MultiClassPerceptron__classes_percetron[1].nb_iter()

)

cross_valisation_mcp_res_versicolor = list(map(mcp_iris_classifier.predict, cross_validation_list_iris_versicolor))

pprint(cross_valisation_mcp_res_versicolor)



print("\nclass 3:")

print(

    "nb training case used : ",

    mcp_iris_classifier._MultiClassPerceptron__classes_percetron[2].nb_iter()

)

cross_valisation_mcp_res_virginica = list(map(mcp_iris_classifier.predict, cross_validation_list_iris_virginica))

pprint(cross_valisation_mcp_res_virginica)
stacked_mcp = MultiClassPerceptron(3, 3)

stacked_mcp.fit(

    [

        cross_valisation_mcp_res_setosa,

        cross_valisation_mcp_res_versicolor,

        cross_valisation_mcp_res_virginica

    ],

    500)
print("class 1:")

testing_p1_res_setosa = list(map(mcp_iris_classifier.predict, testing_list_iris_setosa))

#pprint(testing_p1_res_setosa)

testing_p2_res_setosa = list(map(stacked_mcp.predict, testing_p1_res_setosa))

pprint(testing_p2_res_setosa)



print("\nclass 2:")

testing_p1_res_versicolor = list(map(mcp_iris_classifier.predict, testing_list_iris_versicolor))

#pprint(testing_p1_res_versicolor)

testing_p2_res_versicolor = list(map(stacked_mcp.predict, testing_p1_res_versicolor))

pprint(testing_p2_res_versicolor)



print("\nclass 3:")

testing_p1_res_virginica = list(map(mcp_iris_classifier.predict, testing_list_iris_virginica))

#pprint(testing_p1_res_virginica)

testing_p2_res_virginica = list(map(stacked_mcp.predict, testing_p1_res_virginica))

pprint(testing_p2_res_virginica)
testing_p1_res_setosa     = np.array(testing_p1_res_setosa)

testing_p1_res_versicolor = np.array(testing_p1_res_versicolor)

testing_p1_res_virginica  = np.array(testing_p1_res_virginica)



testing_p2_res_setosa     = np.array(testing_p2_res_setosa)

testing_p2_res_versicolor = np.array(testing_p2_res_versicolor)

testing_p2_res_virginica  = np.array(testing_p2_res_virginica)
print("Setosa class")



#accuracy p1 setosa

p1_tp_setosa = sum(testing_p1_res_setosa[:,0]) / float(len(testing_p1_res_setosa[:,0]))

p1_tn_setosa = (

    (sum(testing_p1_res_versicolor[:,0] == False) + sum(testing_p1_res_virginica[:,0] == False)) /\

    (float(len(testing_p1_res_versicolor[:,0])) + float(len(testing_p1_res_virginica[:,0])))

)

accuracy_p1_setosa = (p1_tp_setosa + p1_tn_setosa) / 2.



#accuracy p2 setosa

p2_tp_setosa = sum(testing_p2_res_setosa[:,0]) / float(len(testing_p2_res_setosa[:,0]))

p2_tn_setosa = (

    (sum(testing_p2_res_versicolor[:,0] == False) + sum(testing_p2_res_virginica[:,0] == False)) /\

    (float(len(testing_p2_res_versicolor[:,0])) + float(len(testing_p2_res_virginica[:,0])))

)

accuracy_p2_setosa = (p2_tp_setosa + p2_tn_setosa) / 2.



print("accuracy class 1, with 1 perceptron  :", accuracy_p1_setosa)

print("accuracy class 1, with 2 perceptrons :", accuracy_p2_setosa)
print("Versicolor class")



#accuracy p1 versicolor

p1_tp_versicolor = sum(testing_p1_res_versicolor[:,1]) / float(len(testing_p1_res_versicolor[:,1]))

p1_tn_versicolor = (

    (sum(testing_p1_res_setosa[:,1] == False) + sum(testing_p1_res_virginica[:,1] == False)) /\

    (float(len(testing_p1_res_setosa[:,1])) + float(len(testing_p1_res_virginica[:,1])))

)

accuracy_p1_versicolor = (p1_tp_versicolor + p1_tn_versicolor) / 2.



#accuracy p2 versicolor

p2_tp_versicolor = sum(testing_p2_res_versicolor[:,1]) / float(len(testing_p2_res_versicolor[:,1]))

p2_tn_versicolor = (

    (sum(testing_p2_res_setosa[:,1] == False) + sum(testing_p2_res_virginica[:,1] == False)) /\

    (float(len(testing_p2_res_setosa[:,1])) + float(len(testing_p2_res_virginica[:,1])))

)

accuracy_p2_versicolor = (p2_tp_versicolor + p2_tn_versicolor) / 2.



print("accuracy class 2, with 1 perceptron  :", accuracy_p1_versicolor)

print("accuracy class 2, with 2 perceptrons :", accuracy_p2_versicolor)
print("Virginica class")



#accuracy p1 virginica

p1_tp_virginica = sum(testing_p1_res_virginica[:,2]) / float(len(testing_p1_res_virginica[:,2]))

p1_tn_virginica = (

    (sum(testing_p1_res_setosa[:,2] == False) + sum(testing_p1_res_versicolor[:,2] == False)) /\

    (float(len(testing_p1_res_setosa[:,2])) + float(len(testing_p1_res_versicolor[:,2])))

)

accuracy_p1_virginica = (p1_tp_virginica + p1_tn_virginica) / 2.



#accuracy p2 virginica

p2_tp_virginica = sum(testing_p2_res_virginica[:,2]) / float(len(testing_p2_res_virginica[:,2]))

p2_tn_virginica = (

    (sum(testing_p2_res_setosa[:,2] == False) + sum(testing_p2_res_versicolor[:,2] == False)) /\

    (float(len(testing_p2_res_setosa[:,2])) + float(len(testing_p2_res_versicolor[:,2])))

)

accuracy_p2_virginica = (p2_tp_versicolor + p2_tn_virginica) / 2.



print("accuracy class 3, with 1 perceptron  :", accuracy_p1_virginica)

print("accuracy class 3, with 2 perceptrons :", accuracy_p2_virginica)
mcp_1_acc = (accuracy_p1_setosa + accuracy_p1_versicolor + accuracy_p1_virginica) / 3.

mcp_2_acc = (accuracy_p2_setosa + accuracy_p2_versicolor + accuracy_p2_virginica) / 3.



print("Total MCPerceptron1 accuracy :", mcp_1_acc)

print("Total MCPerceptron2 accuracy :", mcp_2_acc)

print("Improvment :", (mcp_2_acc - mcp_1_acc) * 100, "%")
testing_p1_res_setosa     = np.array(testing_p1_res_setosa)

testing_p1_res_versicolor = np.array(testing_p1_res_versicolor)

testing_p1_res_virginica  = np.array(testing_p1_res_virginica)



testing_p2_res_setosa     = np.array(testing_p2_res_setosa)

testing_p2_res_versicolor = np.array(testing_p2_res_versicolor)

testing_p2_res_virginica  = np.array(testing_p2_res_virginica)
print("Setosa class")



#accuracy p1 setosa

p1_tp_setosa = sum(testing_p1_res_setosa[:,0]) / float(len(testing_p1_res_setosa[:,0]))

p1_tn_setosa = (

    (sum(testing_p1_res_versicolor[:,0] == False) + sum(testing_p1_res_virginica[:,0] == False)) /\

    (float(len(testing_p1_res_versicolor[:,0])) + float(len(testing_p1_res_virginica[:,0])))

)

accuracy_p1_setosa = (p1_tp_setosa + p1_tn_setosa) / 2.



#accuracy p2 setosa

p2_tp_setosa = sum(testing_p2_res_setosa[:,0]) / float(len(testing_p2_res_setosa[:,0]))

p2_tn_setosa = (

    (sum(testing_p2_res_versicolor[:,0] == False) + sum(testing_p2_res_virginica[:,0] == False)) /\

    (float(len(testing_p2_res_versicolor[:,0])) + float(len(testing_p2_res_virginica[:,0])))

)

accuracy_p2_setosa = (p2_tp_setosa + p2_tn_setosa) / 2.



print("accuracy class 1, with 1 perceptron  :", accuracy_p1_setosa)

print("accuracy class 1, with 2 perceptrons :", accuracy_p2_setosa)
print("Versicolor class")



#accuracy p1 versicolor

p1_tp_versicolor = sum(testing_p1_res_versicolor[:,1]) / float(len(testing_p1_res_versicolor[:,1]))

p1_tn_versicolor = (

    (sum(testing_p1_res_setosa[:,1] == False) + sum(testing_p1_res_virginica[:,1] == False)) /\

    (float(len(testing_p1_res_setosa[:,1])) + float(len(testing_p1_res_virginica[:,1])))

)

accuracy_p1_versicolor = (p1_tp_versicolor + p1_tn_versicolor) / 2.



#accuracy p2 versicolor

p2_tp_versicolor = sum(testing_p2_res_versicolor[:,1]) / float(len(testing_p2_res_versicolor[:,1]))

p2_tn_versicolor = (

    (sum(testing_p2_res_setosa[:,1] == False) + sum(testing_p2_res_virginica[:,1] == False)) /\

    (float(len(testing_p2_res_setosa[:,1])) + float(len(testing_p2_res_virginica[:,1])))

)

accuracy_p2_versicolor = (p2_tp_versicolor + p2_tn_versicolor) / 2.



print("accuracy class 2, with 1 perceptron  :", accuracy_p1_versicolor)

print("accuracy class 2, with 2 perceptrons :", accuracy_p2_versicolor)
print("Virginica class")



#accuracy p1 virginica

p1_tp_virginica = sum(testing_p1_res_virginica[:,2]) / float(len(testing_p1_res_virginica[:,2]))

p1_tn_virginica = (

    (sum(testing_p1_res_setosa[:,2] == False) + sum(testing_p1_res_versicolor[:,2] == False)) /\

    (float(len(testing_p1_res_setosa[:,2])) + float(len(testing_p1_res_versicolor[:,2])))

)

accuracy_p1_virginica = (p1_tp_virginica + p1_tn_virginica) / 2.



#accuracy p2 virginica

p2_tp_virginica = sum(testing_p2_res_virginica[:,2]) / float(len(testing_p2_res_virginica[:,2]))

p2_tn_virginica = (

    (sum(testing_p2_res_setosa[:,2] == False) + sum(testing_p2_res_versicolor[:,2] == False)) /\

    (float(len(testing_p2_res_setosa[:,2])) + float(len(testing_p2_res_versicolor[:,2])))

)

accuracy_p2_virginica = (p2_tp_versicolor + p2_tn_virginica) / 2.



print("accuracy class 3, with 1 perceptron  :", accuracy_p1_virginica)

print("accuracy class 3, with 2 perceptrons :", accuracy_p2_virginica)
mcp_1_acc = (accuracy_p1_setosa + accuracy_p1_versicolor + accuracy_p1_virginica) / 3.

mcp_2_acc = (accuracy_p2_setosa + accuracy_p2_versicolor + accuracy_p2_virginica) / 3.



print("Total MCPerceptron1 accuracy :", mcp_1_acc)

print("Total MCPerceptron2 accuracy :", mcp_2_acc)

print("Improvment :", (mcp_2_acc - mcp_1_acc) * 100, "%")