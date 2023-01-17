mile = float(input('Please enter miles:'))
km = mile /0.62137 
meter = 1000*km
print('miles is equivalent to' + str(round(km,2)))
print('mile is equivalent to' + str(round(meter,2)))
Name = input('Please input your name:')
Age = int(input('Please input your age:'))
Age_in_year_2047 = Age + 2047 - 2020
print('Hi', Name, '!In 2047 you will be', str(Age_in_year_2047),'!')
