# инициализация переменных

summa = 0

lst = []



STOP = 300

ITERATIONS = 10000000



#запуск цикла

for _ in range(ITERATIONS):

    inpt = int(input('Введите целое число: '))

    

    lst.append(inpt)

    summa += inpt

    

    print('Сумма = ', summa)

    print('Количество элементов = ', len(lst))

    

    if summa > STOP:

        break
# инициализация переменных

summa = 0

lst = []



STOP = 100

ITERATIONS = 10000000



#запуск цикла

for _ in range(ITERATIONS):

    inpt = int(input('Введите целое число: '))

    

    if inpt > 30:

        print('Введенное число больше 30')

        continue

    

    lst.append(inpt)

    summa += inpt

    

    print('Сумма = ', summa)

    print('Количество элементов = ', len(lst))

    

    if summa > STOP:

        break
# инициализация переменных

summa = 0

lst = []



STOP = 100



while summa <= STOP:

    inpt = int(input('Введите целое число: '))

    

    if inpt > 30:

        print('Введенное число больше 30')

        continue

    

    lst.append(inpt)

    summa += inpt

    

    print('Сумма = ', summa)

    print('Количество элементов = ', len(lst))
tpl_1 = ()



type(tpl_1)
tpl_2 = tuple()



type(tpl_2)
(1)
type((1))
(1,)
type((1,))
1,
tpl_2 = ('жара', 912, 0.7, True)

tpl_2
tpl_2[0]
tpl_2[0:2]
tpl_2[0] = 3
lst_2 = list(tpl_2)

lst_2
lst_2.__sizeof__()
tpl_2.__sizeof__()
tpl_1 = ('молоко',)

print(tpl_1)



tpl_1 = tuple('молоко')

print(tpl_1)
del tpl_1[0]
del tpl_1
tpl_2
key, value, i, f = tpl_2
print(key)

print(value)

print(i)

print(f)
el_0, el_1, el_2 = tpl_2
tpl_1 = key, value, f

tpl_1
tpl_1 = (('Бананы', 10), ['Огурцы', 20], True)

tpl_1
tpl_1[0:2]
tpl_1[1][1]
tpl_1[1][1] = 12
tpl_1
del tpl_1[1]
del tpl_1[1][0]
tpl_1
tpl_1[1]
tpl_1[1].append('помидор')
tpl_1
tpl_2
tpl_2.count('жара')
tpl_2.count('Зохраб')
tpl_2.index(0.7)
tpl_2.index(12)
for i in tpl_2:

    print(i)
'жара' in tpl_2
'жара' not in tpl_2