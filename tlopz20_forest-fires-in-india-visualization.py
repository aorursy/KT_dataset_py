# Thank you for checking, im a beginner..kindly rate me with a comment.
import pandas as pd

data = pd.read_csv('/kaggle/input/forest-fires-in-india/datafile.csv')

data.head()
data.info()
#lets visualise srates with the highest number of forest fires.
T10 = data.sort_values('2010-2011', ascending = False)
import matplotlib.pyplot as plt

N1 = T10.plot('States/UTs','2010-2011', kind = 'bar',figsize = (20,8), width = 0.8, color = 'red')

plt.title("Histogram showing forest fire occurence in states India from 2010 to 2011")

plt.ylabel("Number of occurence")

plt.xlabel("States")
#FROM THE CHART ABOVE WE CAN SEE THAT MIZORAM HAS THE HIGHEST NUMBER OF FOREST FIRE IN 2010-2011 FOLLOWED BY MADHYA-PRADESH
T09 = data.sort_values('2009-10', ascending = False)

N2 = T09.plot('States/UTs','2009-10', kind = 'bar',figsize = (20,8), width = 0.8, color = 'grey')

plt.title("Histogram showing forest fire occurence in states India from 2009 to 2010")

plt.ylabel("Number of occurence")

plt.xlabel("States")
#FROM THE CHART ABOVE WE CAN SEE THAT MIZORAM HAS THE HIGHEST NUMBER OF FOREST FIRE IN 2009-2010 FOLLOWED BY CHHATISGARH
T08 = data.sort_values('2008-09', ascending = False)

N3 = T08.plot('States/UTs','2008-09', kind = 'bar',figsize = (20,8), width = 0.8, color = 'gold')

plt.title("Histogram showing forest fire occurence in states India from 2008 to 2009")

plt.ylabel("Number of occurence")

plt.xlabel("States")
#FROM THE CHART ABOVE WE CAN SEE THAT MIZORAM HAS THE HIGHEST NUMBER OF FOREST FIRE IN 2009-08 FOLLOWED BY MADYA-PRADESH
data['Total']= data['2010-2011'] + data['2009-10'] + data['2008-09']

data.head()
TT = data.sort_values('Total', ascending = False)

TTT = TT.plot('States/UTs','Total', kind = 'bar',figsize = (20,8), width = 0.8, color = 'orange')

plt.title("Histogram showing Total forest fire occurence in states India from 2008 to 2011")

plt.ylabel("Number of occurence")

plt.xlabel("States")
#FROM THE CHART ABOVE WE CAN SEE THAT MIZORAM HAS THE HIGHEST NUMBER OF FOREST FIRE FOR THE TOTAL 3 YEARS FOLLOWED BY CHHATISGARH
# TO GET THE MEAN VALUE OF FIRE OCCURENCES OVER THE YEARS

dm = data.mean()

mdf = pd.DataFrame(dm)

mdf.rename(columns={"YY" : "mean"}, inplace = True)

mdf.drop('Total')

mdf
mdf.plot(figsize = (20,8))

plt.title("Flow of forest fires over the years from 2008-2011")

plt.ylabel("Mean number occurences of states")

plt.xlabel("Years")
#FROM THIS CHART WE CAN SEE THAT FIRE OCCURENCES INCREASES FROM 2008-2009 TO 2009-10 THEN A FALL IN 2010-2011
labels = '2010-2011', '2009-10', '2008-09'

sizes = [397.085714, 882.628571, 694.514286]

colors = ['Red', 'yellowgreen', 'lightcoral']

explode = (0.1, 0, 0)  # explode 1st slice



# Plot

plt.pie(sizes, explode=explode, labels=labels, colors=colors,

autopct='%1.1f%%', shadow=True, startangle=140)



plt.axis('equal')

plt.show()
#The above pie chart shows the year with most fire occurences.