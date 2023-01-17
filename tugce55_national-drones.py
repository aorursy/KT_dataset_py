# Our national tactical and male drones.



drones=['bayraktar_tb2','anka','akıncı','bayraktar_mini_iha','karayel','aksungur','kargu']
print(drones)
print("Type of 'drones' is : " , type(drones))
# The companies of these drones.
companies=['baykar','tai','vestel','stm']
print(companies)
tactical = drones[0],drones[2],drones[3],drones[5],drones[6]

print(tactical)

print("Type of 'v_list2_4' is : " , type(tactical))
male=drones[1]

print(male)
observation=drones[-3]

print(observation)
#inventory



inventory=drones[0:2],drones[3],drones[-2]

print('Our Army has ',inventory)

print('The type is:',type(inventory))

#Len

v_len_drones = len(drones)

v_len_componies= len(companies)

print("Size of 'drones' is : ",v_len_drones)

print('Size of componies is:',v_len_componies)

#Those to be appended until 2023



drones.append('akıncı,kargu,and more')

print(drones)



companies.append('Gokturk Technology (it will be also my company),and more')

print(companies)
#Reverse

drones.reverse()

print(drones)

#Sort

drones.sort()

print(drones)