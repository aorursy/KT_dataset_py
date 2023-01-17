#Finger Exercice 1
#Author : Yahya KABBOUL

Mile = float(input("Please enter the distance in Miles that you want to convert :"))  #Asking miles input from the user

#Formulas for conversion :

Mile_km = Mile * 0.62137  
Mile_meter = 1000 * Mile_km

#Showing the output as a result of conversions and limiting the numbers after . to 2 digits 

print('=> {:.2f} miles are equal to {:.2f} km & {:.2f} meters' .format(Mile,Mile_km,Mile_meter))
#Finger Exercice 2
#Author : Yahya KABBOUL

Name = str(input("What's your name please ? :"))  #Asking the user's name
print('Welcome to our Program',Name,)

Age = int(input("What's your age please ? :"))    #Asking the user's age

Age_in_year_2047 = Age + 2047 - 2020  #Formula to calculate the futur age using the input age number

#Showing the user's age in 2047 according the his current age

print ('Hi ',Name,'! In 2047 you will be',Age_in_year_2047,'!')
#Finger Exercice 2 Bonus
#Author : Yahya KABBOUL

Name = str(input("What's your name please ? :"))  #Asking the user's name
print('Welcome to our Program',Name,)

Age = int(input("What's your age please ? :"))    #Asking the user's age

Year = int(input("Please enter a year in witch you want to see your age ? :"))    #Asking a year in which the user want to know the age

Age_in_year_x = Age + Year - 2020

#Showing the user's age according to the inputs (Age+Predited Year)

print ('Hi ',Name,'! In',Year,' you will be',Age_in_year_x,'!')