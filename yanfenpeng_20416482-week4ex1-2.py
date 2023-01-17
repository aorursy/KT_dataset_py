mile_a = input("please enter miles:") #ask user enter calculated mile

mile_a = float(mile_a) #str format convert to float format

km_a = mile_a / 0.62137

meter_a = mile_a * 1000

print("%.2f miles is equivalent to" %(float(mile_a))) #keep two decimal places

print("%.2f km "%(float(km_a))+"/"+"%.2f meters"%(float(meter_a))) #keep two decimal places
import datetime as dt #import datetime module



UserName = input("please enter your name:") #accept user's name

UserAge = int(input("please enter your age:")) #accept user's age and convert to int format

FutherAge = 2047 - int(dt.datetime.today().year) + UserAge

#get the current year of system and calculate age

print("HI!%s!"%(UserName)+"In 2047, you will be %s!"%(FutherAge))