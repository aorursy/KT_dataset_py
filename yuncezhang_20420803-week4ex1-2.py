#Finger Exercise 1
mile = input("Enter miles")

mile = float(mile)

km = mile/0.62137

meters = 1000 * km

mile_km = mile/0.62137

mile_meter = 1000*mile_km

print('{:.2f} miles is equivalent to'.format(mile))

print('{:6.4f}km/{:6.1f}meters'.format(mile_km,mile_meter))


#Finger Exercise 2
name=input('enter your name:')

age=input('enter your age:')

#name = str(name)

age=int(age)

age_in_year_2047=age+2047-2020

print('Hi {}! In 2047 you will be {:d}!'.format(name,age_in_year_2047))