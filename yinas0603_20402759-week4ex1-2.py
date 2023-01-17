mile=float(input("Enter miles: "))
km=mile/0.62137
meters=1000*km
mile_km=mile/0.62137
mile_meter=1000*mile_km
print('{:.2f}miles is equivalent to'.format(mile))
print('{:6.4f}km/{:6.1f}meters'.format(mile_km,mile_meter))
name = str(input("what is your name: "))
age = int(input("How old are you: "))
age_2047 = age+2047-2020
print('Hi '+str(name)+'！In 2047 you will be '+str(age_2047)+'!') # type error: can only concatenate str (not "int")to str, print(1+'a') revise to print (str(1)+'a')