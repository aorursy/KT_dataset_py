import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("../input/train.csv")

df.head()
survived = len([x for x in df["Survived"] if x == 1 ])

died = len([x for x in df["Survived"] if x == 0 ])

print("Total:", df.shape[0], "Survived: ", survived, " Died:", died)



x = ["Survived", "Died"]

xy = [survived, died]



plt.bar(range(len(x)), xy, color=["blue","green"]) 

plt.xticks(range(len(x)),x)

plt.title("How much people die ?")
womens_dead = [ x for x in df["Sex"] if x == "female"]

mens_dead = [ x for x in df["Sex"] if x == "male"]



graphic_labels = ["Women", "Men"]

xy = [len(womens_dead), len(mens_dead)]



plt.bar(range(len(graphic_labels)), xy, color=["pink", "black"])

plt.xticks(range(len(graphic_labels)), graphic_labels)

plt.title("who died more? women or men ?")
haveSibSp = [x for x in df["SibSp"] if x > 0 and x]


