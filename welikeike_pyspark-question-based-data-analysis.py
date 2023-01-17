!pip install pyspark

from pyspark.sql import SparkSession

import pandas as pd
#initialization of spark session

sk = SparkSession.builder.appName("Project").getOrCreate()



#load the datasets

df = sk.read.format("csv").option("header", "true").load("../input/summer-olympics-medals/Summer-Olympic-medals-1976-to-2008.csv")

exceptions = sk.read.format("csv").option("header","true").load("../input/didnotparticipate1976to2008/Did-Not-Participate-1976-to-2008.csv")
#Structure of the datasets

df.printSchema()

exceptions.printSchema()
%%time

#Q1: Which country won the most gold medals?



#The only useful columns are "Country" and "Medal" so we select them

imp = df.rdd.map(lambda x: (x[8],x[10]))

#We are only interested in gold medals so we filter the others out, we then create tuples ("Country",number of gold medals)

med = imp.filter(lambda x: x[1] == "Gold").map(lambda x: (x[0],1)).reduceByKey(lambda acc,x: x + acc)

#Finally we sort the result in descending order

result = med.map(lambda x: (x[1],x[0])).sortByKey(0).collect()
for e in result:

    print(e)
%%time

#Q2: What disciplines are mixed (both male and female compete against each other)?



#Select the columns containing discipline and gender of the event

imp = df.rdd.map(lambda x: (x[3],x[9]))

#In the "Event_gender" column an "X" marks mixed events

med = imp.filter(lambda x: x[1] == "X").groupByKey()

#Return the name of the disciplines with at least one mixed event

result = med.map(lambda x: x[0]).collect()
for e in result:

    print(e)
%%time

#Q3: In mixed sports, are the gold medals winners composed of more male or female athletes?



#Select the columns containing gender of the athlete, gender of the events and medals

imp = df.rdd.map(lambda x: (x[6],x[9],x[10]))

#Keep only mixed events and count the number of gold medals for the 2 genders

med = imp.filter(lambda x: x[1] == "X" and x[2] == "Gold").map(lambda x: (x[0],1)).reduceByKey(lambda acc,x: x + acc)

#Sort and format result

result = med.map(lambda x: (x[1],x[0])).sortByKey(0).map(lambda x: (x[1],x[0])).collect()
for e in result:

    print(e)
%%time

#Q4: Which disciplines are only for man or only for women?



#Select disciplines and gender of events

imp = df.rdd.map(lambda x: (x[3],x[9]))

#Keep the disciplines with male events with no repetitions

med = imp.filter(lambda x: x[1] == "M").map(lambda x: ((x[0]),x[1])).distinct()

#Keep the disciplines with female events with no repetitions

med2 = imp.filter(lambda x: x[1] == "W").map(lambda x: ((x[0]),x[1])).distinct()

#Join the 2 results and keep only the disciplines that are either men only or women only

med3 = med.fullOuterJoin(med2).filter(lambda x: x[1][0] is None or x[1][1] is None)

#Format the result

r1 = med3.filter(lambda x: x[1][1] is None).map(lambda x: ("Men Only",x[0])).reduceByKey(lambda acc,x: x + ", " + acc).collect()

r2 = med3.filter(lambda x: x[1][0] is None).map(lambda x: ("Women Only",x[0])).reduceByKey(lambda acc,x: x + ", " + acc).collect()

result = [r1,r2]
for e in result:

    print(e)
%%time

#Q5: Top 3 countries which gained most and least medals in team sports?



#Select all the columns except name and gender of the athlete (the only 2 things that differentiate between teammates)

imp = df.rdd.map(lambda x: (x[0],x[1],x[2],x[3],x[4],x[7],x[8],x[9],x[10]))

#Sum the number of team members for each winning team

med = imp.map(lambda x: ((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),1)).reduceByKey(lambda acc,x: x + acc)

#For each country sum the number of team members after removing the teams with only one member (single athlete sports)

med2 = med.map(lambda x: (x[0][6],x[1])).filter(lambda x: x[1]>1).reduceByKey(lambda acc,x: x + acc).map(lambda x: (x[1],x[0]))

#Sort the number of won medals and keep only the 3 best and worst results

r1 = med2.sortByKey(0).take(3)

r2 = med2.sortByKey(1).take(3)

result = [r1,r2]
for e in result:

    print(e)
%%time

#Q6: Which countries didn't gain any medals from team sports?



#Select all the columns except name and gender of the athlete (the only 2 things that differentiate between teammates) 

imp = df.rdd.map(lambda x: (x[0],x[1],x[2],x[3],x[4],x[7],x[8],x[9],x[10]))

#Sum the number of team members for each winning team

med = imp.map(lambda x: ((x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]),1)).reduceByKey(lambda acc,x: x + acc)

#For each country sum the number of team members after removing the teams with only one member (single athlete sports) 

med2 = med.map(lambda x: (x[0][6],x[1])).filter(lambda x: x[1]>1).reduceByKey(lambda acc,x: x + acc)

#Create the list of all the countries in the dataset

ref = df.rdd.map(lambda x: (x[8],"")).distinct()

#Return only the countries that won no medals for team sports

result = ref.leftOuterJoin(med2).filter(lambda x: x[1][1] is None).map(lambda x: x[0]).collect()
for e in result:

    print(e)
%%time

#Q7: Which country won more medals for each discipline?



#Select the columns containing disciplines, countries and medals

imp = df.rdd.map(lambda x: (x[3],x[8],x[10]))

#Sum the number of medals for each country for each discipline

med = imp.map(lambda x: ((x[0],x[1]),1)).reduceByKey(lambda acc,x: x + acc)

#Keep only the countries that won the most medals for each discipline

med2 = med.map(lambda x: (x[0][0],(x[0][1],x[1]))).reduceByKey(lambda acc,x: x if x[1] > acc[1] else acc)

#Format the result

result = med2.map(lambda x: (x[0],x[1][0],x[1][1])).collect()
for e in result:

    print(e)
%%time

#Q8: Which country won the most gold medals the year it hosted the olympics?



#First we need to create a table that contains to which country does the city where the olympics took place belong

lookup = sk.sparkContext.parallelize([("Montreal","Canada"),("Moscow","Soviet Union"),("Los Angeles","United States"),("Seoul","Korea, South"),("Barcelona","Spain"),("Atlanta","United States"),("Sydney","Australia"),("Athens","Greece"),("Beijing","China")])

#Select the columns containing the olympic city, country and medals

imp = df.rdd.map(lambda x: (x[0],x[8],x[10]))

#Keep only the gold medals and group them for each country for each city

med = imp.filter(lambda x: x[2] == "Gold").map(lambda x: ((x[0],x[1]),1)).reduceByKey(lambda acc,x: x + acc)

#Keep only the olympics where each country won the most gold medals

med2 = med.map(lambda x: (x[0][1],(x[0][0],x[1]))).reduceByKey(lambda acc,x: x if x[1] > acc[1] else acc)

#Check with the table we created if a country won the most gold medals when the city hosting the olympics belonged to that country

med3 = med2.map(lambda x: (x[1][0],(x[0],x[1][1]))).join(lookup).filter(lambda x: x[1][0][0] == x[1][1])

#Format the result

result = med3.map(lambda x: (x[0],x[1][1],x[1][0][1])).collect()
for e in result:

    print(e)
%%time

#Q9: Which athletes won at least a medal in most olympics?



#Select name of the athlete and olympic year without repetitions (more than one medal won in that year)

imp = df.rdd.map(lambda x: (x[5],x[1])).distinct()

#Calculate in how many olympics did the athlete win a medal

med = imp.map(lambda x: (x[0],1)).reduceByKey(lambda acc,x: x + acc).map(lambda x: (x[1],x[0]))

#Sort the result

result = med.sortByKey(0).collect()
for e in result:

    print(e)
%%time

#Q10: What is the average number of medals that each country won?



#Select the columns containing the countries and the year of the olympics

imp = df.rdd.map(lambda x: (x[8],x[1]))

#Sum up the number of medals for each country for each year

med = imp.map(lambda x: ((x[0],x[1]),1)).reduceByKey(lambda acc,x: x + acc)

#Create the list of all the olympics in the dataset

ref = df.rdd.map(lambda x: (x[1])).distinct()

#Create the list of all the countries in the dataset, then cartesian product them together with the list of olympics

ref2 = df.rdd.map(lambda x: (x[8])).distinct().cartesian(ref).map(lambda x: ((x[0],x[1]),""))

#From the result remove the couple (country, year) if the country didn't participate that year

ref3 = exceptions.rdd.map(lambda x: ((x[0],x[1]),"")).rightOuterJoin(ref2).filter(lambda x: x[1][0] is None).map(lambda x: (x[0],""))

#If the country did participate but didn't have any medals for that year it should count as having "won" 0 medals

med2 = ref3.leftOuterJoin(med).map(lambda x: (x[0][0],x[1][1]) if x[1][1] is not None else (x[0][0],0))

#Calculate the average

med3 = med2.aggregateByKey((0,0),lambda a,b: (a[0]+b, a[1]+1),lambda a,b: (a[0]+b[0],a[1]+b[1])).mapValues(lambda x: x[0]/x[1])

#Sort and format the result

result = med3.map(lambda x: (x[1],x[0])).sortByKey(0).collect()
#Final result

for e in result:

    print(e)
sk.stop()