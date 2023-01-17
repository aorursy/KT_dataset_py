import pandas as pd                       

import matplotlib.pyplot as plt           

data=pd.read_csv('../input/iris/Iris.csv') 
print(data.info()) #information on datatype, memory usage, null or non null values

print(data.describe()) #descriptive statistics

print(data.head())     #First few parts of the data
data.Species.value_counts()
Target_counts = data.Species.value_counts()

Target_labels = data.Species.unique()

plt.pie(Target_counts,explode = [0.1,0.1,0.1],labels = Target_labels)
plt.bar(Target_labels,Target_counts)

plt.ylabel('Counts')

plt.title('Distribution of the Target variable')
data.drop('Id',axis=1,inplace=True)



print(data.head())
#Trifurcating data according to there species

df1 = data[data.Species=='Iris-setosa']

df2 = data[data.Species=='Iris-versicolor']

df3 = data[data.Species=='Iris-virginica']
#plotting sepal Length vs width for different species in the same plot 

plt.plot(df1.SepalLengthCm,df1.SepalWidthCm,linestyle='none',marker='o',c='red',label='setosa')

plt.plot(df2.SepalLengthCm,df2.SepalWidthCm,linestyle='none',marker='o',c='green',label='versicolor')

plt.plot(df3.SepalLengthCm,df3.SepalWidthCm,linestyle='none',marker='o',c='blue',label='virginica')



#setting title,labels,legend

plt.title('Sepal Width and Length')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Sepal Width (cm)')



plt.legend()



plt.show() 
#This is a simplified version of above cell

plt.plot(df1.SepalLengthCm,df1.SepalWidthCm,'ro')

plt.plot(df2.SepalLengthCm,df2.SepalWidthCm,'go')

plt.plot(df3.SepalLengthCm,df3.SepalWidthCm,'bo')



#Note that this can be further simplified by using seaborn, 

#but this notebook is for strictly using matplotlib   

plt.title('Sepal Width and Length')

plt.xlabel('Sepal Length (cm)')

plt.ylabel('Sepal Width (cm)')



plt.legend(['setosa','versicolor','virginica'])



plt.show()
plt.plot(df1.PetalLengthCm,df1.PetalWidthCm,'ro')

plt.plot(df2.PetalLengthCm,df2.PetalWidthCm,'go')

plt.plot(df3.PetalLengthCm,df3.PetalWidthCm,'bo')



plt.title('Petal Width and Length')

plt.xlabel('Petal Length (cm)')

plt.ylabel('Petal Width (cm)')



plt.legend(['setosa','versicolor','virginica'])



plt.show()
plt.boxplot([df1.SepalLengthCm,df2.SepalLengthCm,df3.SepalLengthCm])

plt.title('Sepal Length')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.xlabel('Species')

plt.ylabel('Sepal Length in cm')
plt.boxplot([df1.SepalWidthCm,df2.SepalWidthCm,df3.SepalWidthCm])

plt.title('Sepal Width')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.xlabel('Species')

plt.ylabel('Sepal Width in cm')
plt.subplot(121)



plt.boxplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm])

plt.title('Petal Length')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.xlabel('Species')

plt.ylabel('Petal Length in cm')



plt.subplot(122)



plt.boxplot([df1.PetalWidthCm,df2.PetalWidthCm,df3.PetalWidthCm])

plt.title('Petal Width')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.xlabel('Species')

plt.ylabel('Petal Width in cm')



plt.tight_layout()



plt.show()
# Classification threshold

threshold=max(df1.PetalLengthCm)

print(threshold)
# we will define a function here to return whether the flower is setosa or not

def predict(length):

    if length<=1.9:

        return True

    else:

        return False    
#data is the dataframe containing the complete dataset

twoclass=data[['PetalLengthCm','Species']]
# We store whether the Species is setosa or not from the known Species column.

# This is the actual value

actual=twoclass.Species=='Iris-setosa'

#The Species are predicted from petal length

predicted=twoclass.PetalLengthCm.apply(predict)
accuracy=sum(actual==predicted)/len(actual)*100

print('accuracy is {}%'.format(accuracy))
#Visualizing the result

plt.plot(twoclass.PetalLengthCm,predicted,'bo')

plt.title('Setosa or not')

plt.xlabel('Petal length in cm')

plt.yticks([0,1],['Not setosa','setosa'])

plt.vlines(1.9,0,1,colors='red') 
#violin plot

plt.figure(figsize=(15,17))



plt.subplot(221)

plt.violinplot([df1.SepalLengthCm,df2.SepalLengthCm,df3.SepalLengthCm],showmedians=True)

plt.xlabel('Species')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.ylabel('Sepal Length in cm')

plt.title('Sepal Length')



plt.subplot(222)

plt.violinplot([df1.SepalWidthCm,df2.SepalWidthCm,df3.SepalWidthCm],showmedians=True)

plt.xlabel('Species')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.ylabel('Sepal Width in cm')

plt.title('Sepal Width')



plt.subplot(223)

plt.violinplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm],showmedians=True)

plt.xlabel('Species')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.ylabel('Petal Length in cm')

plt.title('Petal Length')



plt.subplot(224)

plt.violinplot([df1.PetalWidthCm,df2.PetalWidthCm,df3.PetalWidthCm],showmedians=True)

plt.xlabel('Species')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.ylabel('Petal Width in cm')

plt.title('Petal Width')



plt.tight_layout()
#histogram

plt.figure(figsize=(15,15))

plt.subplot(221)

plt.hist(data.SepalLengthCm,bins=30)

plt.title('Sepal Length')



plt.subplot(222)

plt.hist(data.SepalWidthCm,bins=30)

plt.title('Sepal Width')



plt.subplot(223)

plt.hist(data.PetalLengthCm,bins=30)

plt.title('Petal Length')



plt.subplot(224)

plt.hist(data.PetalWidthCm,bins=30)

plt.title('Petal Width')
#histogram

plt.figure(figsize=(15,15))

plt.subplot(221)

plt.hist(df1.SepalLengthCm,bins=30)

plt.hist(df2.SepalLengthCm,bins=30)

plt.hist(df3.SepalLengthCm,bins=30)

plt.legend(['Setosa','Versicolor','Virginica'])

plt.title('Sepal Length')



plt.subplot(222)

plt.hist(df1.SepalWidthCm,bins=30)

plt.hist(df2.SepalWidthCm,bins=30)

plt.hist(df3.SepalWidthCm,bins=30)

plt.legend(['Setosa','Versicolor','Virginica'])

plt.title('Sepal Width')



plt.subplot(223)

plt.hist(df1.PetalLengthCm,bins=30)

plt.hist(df2.PetalLengthCm,bins=30)

plt.hist(df3.PetalLengthCm,bins=30)

plt.legend(['Setosa','Versicolor','Virginica'])

plt.title('Petal Length')



plt.subplot(224)

plt.hist(df1.PetalWidthCm,bins=30)

plt.hist(df2.PetalWidthCm,bins=30)

plt.hist(df3.PetalWidthCm,bins=30)

plt.legend(['Setosa','Versicolor','Virginica'])

plt.title('Petal Width')
#plt.style.use('fivethirtyeight')
#plt.style.use('seaborn')
#plt.style.use('default')
Target_counts = data.Species.value_counts()

Target_labels = data.Species.unique()

plt.pie(Target_counts,explode = [0.2,0,0],labels = Target_labels, 

        startangle = 90, autopct = '%1.2f%%', shadow = True, colors = ['red','green','orange'])

plt.axis('equal')

plt.title('Distribution of Species in the Data')

plt.legend(loc='upper right')

plt.savefig('pie.jpg')
plt.bar(Target_labels,Target_counts,

        width=0.5,color=['Yellow','Green','Blue'],edgecolor='red',linewidth=[2,0,0])

plt.title('Distribution of the Target variable')

plt.savefig('bar.jpg')
plt.plot(df1.PetalLengthCm,df1.PetalWidthCm,linestyle='none',c= 'r',marker='o',ms=5,mec='yellow')

plt.plot(df2.PetalLengthCm,df2.PetalWidthCm,linestyle='none',c='g',marker='d',ms=3,alpha=0.5)

plt.plot(df3.PetalLengthCm,df3.PetalWidthCm,linestyle='none',c='b',marker='^',ms=3,alpha=0.5)



plt.title('Petal Width and Length')

plt.xlabel('Petal Length (cm)')

plt.ylabel('Petal Width (cm)')



plt.legend(['setosa','versicolor','virginica'])



plt.savefig('scatter.jpg')
plt.boxplot([df1.PetalLengthCm,df2.PetalLengthCm,df3.PetalLengthCm],notch=True)

plt.title('Petal Length')

plt.xticks([1,2,3],['Setosa','Versicolor','Virginica'])

plt.xlabel('Species')

plt.ylabel('Petal Length in cm')

plt.savefig('boxplot.jpg')
plt.plot(twoclass.PetalLengthCm,predicted,'bo')

plt.title('Setosa or not')

plt.xlabel('Petal length in cm')

plt.yticks([0,1],['Not setosa','setosa'])

plt.axvspan(0,2.0,facecolor='green',alpha=0.3)