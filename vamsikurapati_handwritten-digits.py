from sklearn.datasets import load_digits

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

digits=load_digits()
print(digits.images.shape)
fig, axes = plt.subplots(10, 10, figsize=(8, 8),

                        subplot_kw={'xticks':[], 'yticks':[]},

                        gridspec_kw=dict(hspace=0.1, wspace=0.1))

for i, ax in enumerate(axes.flat):

    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')

    ax.text(0.05, 0.05, str(digits.target[i]),

            transform=ax.transAxes, color='green')

plt.show()
x=digits.data

y=digits.target
x.shape,y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

model = GaussianNB()      

model.fit(x_train, y_train)  

y_model = model.predict(x_test)   

print(accuracy_score(y_test,y_model))
mat=confusion_matrix(y_test,y_model)

sns.heatmap(mat,square=True,annot=True,cbar=False)

plt.xlabel("predicted value")

plt.ylabel("true value")

plt.show()