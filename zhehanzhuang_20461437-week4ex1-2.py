a=int(input('Please enter a number:')) #let user inpur the name
b1=a/0.62137                           #b1 is the km converted from miles
b2=round(b1,4)                         #b1 has lots of decimals, in order to make code clean, we use round to simplify
c=1000*b2                              #convert km to meters
print('{} miles is equivalent to'.format(a))
print('{} km/{} meters'.format(b2,c))
name=input('Please enter your name:')    #let user input the name
age=int(input('Please enter your age:')) #let user input the age
age_2047=age+27                          #need to plus 27 in 2047
print('Hi {}! In 2047 you will be {}!'.format(name,age_2047))