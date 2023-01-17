lst_1 = []



type(lst_1)
lst_2 = list()



type(lst_2)
id(lst_2)
lst_2 = ['жара', 912, 0.7, True]

lst_2
id(lst_2)
id(lst_1)
lst_1 is lst_2
lst_1 is not lst_2
lst_1 = ['молоко']

print(lst_1)



lst_1 = list('молоко')

print(lst_1)
print(lst_2)

print(id(lst_2))

lst_3 = lst_2



print(lst_3)

print(id(lst_3))
lst_2[0] = 11

print(lst_2)

print(lst_3)

lst_2[0] = 'жара'
print(lst_2)

print(id(lst_2))

lst_3 = list(lst_2)

print(lst_3)

print(id(lst_3))
lst_2[0] = 11

print(lst_2)

print(lst_3)

lst_2[0] = 'жара'
lst_3[0]
lst_3[4]
lst_3[-1]
lst_3[-2]
lst_3
lst_3[0:2]
lst_3[2:4]
lst_3[:2]
lst_3[2:]
lst_3[::2]
lst_3[::-2]
x = slice(0, 2, 1)
type(x)
lst_3[x]
lst_3[:]
lst_3 is lst_3[:]
print(lst_2)

print(id(lst_2))

lst_3 = lst_2[:]



print(lst_3)

print(id(lst_3))
lst_2[0] = 11

print(lst_2)

print(lst_3)

lst_2[0] = 'жара'