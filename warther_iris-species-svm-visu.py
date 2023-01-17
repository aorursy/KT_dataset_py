import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

%matplotlib inline



main_df=pd.read_csv('../input/Iris.csv') 
def MultiClasse(data,variable,classe1,val1,classe2,val2,classe3,val3,

                classe4,val4,classe5,val5,classe6,val6,classe7,val7,

                classe8,val8,classe9,val9,classe10,val10):

                    

    i_max=data.shape[0]

    for i in range(0,i_max):

        if data[variable].values[i]==classe1:

            data[variable].values[i]=val1

        elif data[variable].values[i]==classe2:

            data[variable].values[i]=val2

        elif  data[variable].values[i]==classe3:

            data[variable].values[i]=val3

        elif data[variable].values[i]==classe4:

            data[variable].values[i]=val4

        elif  data[variable].values[i]==classe5:

            data[variable].values[i]=val5

        elif  data[variable].values[i]==classe6:

            data[variable].values[i]==val6

        elif  data[variable].values[i]==classe7:

            data[variable].values[i]=val7

        elif  data[variable].values[i]==classe8:

            data[variable].values[i]=val8

        elif  data[variable].values[i]==classe9:

            data[variable].values[i]=val9

        elif data[variable].values[i]==classe10:

            data[variable].values[i]=val10
#### Transform the 'Species' in 3 classes #### 

#### Iris-setosa=1 , Iris-versicolor=2 , Iris-verginica=3 ####



MultiClasse(main_df,'Species','Iris-setosa',1,'Iris-versicolor',2,

                        'Iris-virginica',3,0,0,0,0,0,0,0,0,0,0,0,0,0,0)

                        

print(main_df.head())
#### Define explanatory and target variable #### 



X=main_df.drop(['Species','Id','SepalLengthCm','SepalWidthCm'],axis=1)

y=main_df['Species']



#### Separation in test/train ####



from sklearn import cross_validation



X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,

                              test_size=0.35,random_state=1) 





X_train=X_train.reset_index(drop=True)

X_test=X_test.reset_index(drop=True)

y_train=y_train.reset_index(drop=True)

y_test=y_test.reset_index(drop=True)



y_train=np.array(y_train,dtype=np.float64)

y_test=np.array(y_test,dtype=np.float64)
#### data scale ####





from sklearn.preprocessing import scale



X_train_np=scale(X_train)

X_test_np=scale(X_test)



X_train=pd.DataFrame(X_train_np)

X_test=pd.DataFrame(X_test_np)
from sklearn import svm



clf=svm.SVC()



clf.fit(X_train,y_train)





#### Grid search ####





from sklearn.grid_search import GridSearchCV



param_grid=[{'kernel':['rbf'],'gamma':[0.1,0.5,1],'C':[0.1,0.5,1,10]},

            {'kernel':['poly'],'gamma':[0.1,0.5,1],'C':[0.1,0.5,1,10],'degree':[2,3]},

            {'kernel':['linear'],'C':[0.1,0.5,1,10]},

            {'kernel':['sigmoid'],'C':[0.1,0.5,1,10]}]

            



grid=GridSearchCV(clf, param_grid,cv=3)



grid.fit(X_train,y_train)



BestPara=grid.best_params_



be=grid.best_estimator_



print("Best Para = " , BestPara)





### Prediction ###



predict=be.predict(X_test)



size=predict.shape[0]

print("taille de y_prediction = %i" %size)



from sklearn.metrics import accuracy_score 



a=accuracy_score(y_test,predict)



print("Score final = %f" %a)
### Confusion matrix ###



from sklearn.metrics import confusion_matrix



conf=confusion_matrix(y_test,predict)



print(conf)



### Data visualization



main1=main_df[main_df['Species']==1]

main2=main_df[main_df['Species']==2]

main3=main_df[main_df['Species']==3]



## Sepal



axes=plt.gca()



plt.subplot(131)



plt.plot(main1['SepalLengthCm'],main1['SepalWidthCm'],'gs',label='Species 1')

plt.plot(main2['SepalLengthCm'],main2['SepalWidthCm'],'r^',label='Species 2')

plt.plot(main3['SepalLengthCm'],main3['SepalWidthCm'],'bo',label='Species 3')

plt.title('Species / Sepal' )

plt.axis([3,9,1,5])

plt.xlabel('Sepal Length')

plt.ylabel('Sepal Width')



## Petal



plt.subplot(133)



plt.plot(main1['PetalLengthCm'],main1['PetalWidthCm'],'gs',label='Species 1')

plt.plot(main2['PetalLengthCm'],main2['PetalWidthCm'],'r^',label='Species 2')

plt.plot(main3['PetalLengthCm'],main3['PetalWidthCm'],'bo',label='Species 3')

plt.title('Species / Petal' )

plt.axis([0,9,-0.5,3])

plt.xlabel('Petal Length')

plt.ylabel('Petal Width')





## Species means



print('Sepal mean length for class 1 is:%f' %main1['SepalLengthCm'].mean())

print('Sepal mean length for class 2 is:%f' %main2['SepalLengthCm'].mean())

print('Sepal mean length for class 3 is:%f' %main3['SepalLengthCm'].mean())



print('Petal mean length for class 1 is:%f' %main1['SepalLengthCm'].mean())

print('Petal mean length for class 2 is:%f' %main2['SepalLengthCm'].mean())

print('Petal mean length for class 3 is:%f' %main3['SepalLengthCm'].mean())



### Plot decision boundary ### 



from matplotlib.colors import ListedColormap



resolution=0.02



# setup marker and color map



markers = ('s', 'x', 'o', '^', 'v')

colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

cmap = ListedColormap(colors[:len(np.unique(y))])



# plot the decision surface





x1_min, x1_max = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1

x2_min, x2_max = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1



xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),

                           np.arange(x2_min, x2_max, resolution))



Z = be.predict(np.array([xx1.ravel(), xx2.ravel()]).T)



Z = Z.reshape(xx1.shape)



plt.contourf(xx1, xx2, Z, alpha=0.5,cmap=cmap)

plt.xlim(xx1.min(), xx1.max())

plt.ylim(xx2.min(), xx2.max())





for idx, cl in enumerate(np.unique(y)):

    plt.scatter(x=X_train_np[y_train == cl, 0], y=X_train_np[y_train == cl, 1],

                alpha=0.7,c=cmap(idx),

                marker=markers[idx], label=cl)