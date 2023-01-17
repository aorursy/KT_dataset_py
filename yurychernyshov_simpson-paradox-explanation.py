import numpy as np

from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression
Yellows = [(xi+1,yi+1) for xi,yi in zip(np.random.rand(10),np.random.rand(10))]

Blues   = [(xi+3,yi+3) for xi,yi in zip(np.random.rand(10),np.random.rand(10))]

plt.scatter([i[0] for i in Yellows],[i[1] for i in Yellows],color="y")

plt.scatter([i[0] for i in Blues],[i[1] for i in Blues],color="b")

plt.title("Yellows and Blues")

plt.grid(True)

plt.show()
def fLinearPrediction(xp,yp):

    x=np.array(xp).reshape((-1,1)) #is nedded to place x values from horizontal to vertical

    y=np.array(yp)

    model = LinearRegression()

    model.fit(x,y)

    r=model.score(x,y)

    #print(r)

    #print(model.intercept_)

    #print(model.coef_)

    return model.predict(x)
fig, axe = plt.subplots(figsize=(15,4),ncols=3)

axe[0].scatter([i[0] for i in Yellows],[i[1] for i in Yellows],color="y")

axe[1].scatter([i[0] for i in Blues],[i[1] for i in Blues],color="b")

axe[0].plot([i[0] for i in Yellows], 

            fLinearPrediction([i[0] for i in Yellows],[i[1] for i in Yellows]),color="r")

axe[1].plot([i[0] for i in Blues], 

            fLinearPrediction([i[0] for i in Blues],[i[1] for i in Blues]), color="r")

axe[2].scatter([i[0] for i in Yellows],[i[1] for i in Yellows],color="y")

axe[2].scatter([i[0] for i in Blues],[i[1] for i in Blues],color="b")

Greens=Yellows+Blues

axe[2].plot([i[0] for i in Blues], 

            fLinearPrediction([i[0] for i in Blues],[i[1] for i in Blues]), color="r")

axe[2].plot([i[0] for i in Yellows], 

            fLinearPrediction([i[0] for i in Yellows],[i[1] for i in Yellows]),color="r")

axe[2].plot([i[0] for i in Greens], 

            fLinearPrediction([i[0] for i in Greens],[i[1] for i in Greens]), color="r")

axe[0].grid(True)

axe[1].grid(True)

axe[2].grid(True)

plt.show()