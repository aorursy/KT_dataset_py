dct_1 = {

    'погода':'жара', 

    'давление':912, 

    'осадки в мм':0.7, 

    'штиль':True,

    'температура':['+1','+4','+2','0'],

}
print(dir(dct_1))
dct_1.keys()
dct_1.keys()[3]
tuple(dct_1.keys())[3]
list(dct_1.keys())[3]
dct_1.values()
dct_1.values()[3]
tuple(dct_1.values())[3]
dct_1.items()
tuple(dct_1.items())[3]
dct_1.update({

    'штиль':False

})



dct_1
dct_1['штиль'] = True



dct_1
dct_1.get('погода')
dct_1['погода']
dct_1.get('овощи')
dct_1['овощи']
dct_1.get('овощи', 'Нет в списке ключей')
dct_1
dct_1.pop('погода')
dct_1
dct_1.popitem()
dct_1
dct_1.fromkeys(['погода', 'штиль','температура'])
dct_1.fromkeys(

    ['погода', 'штиль'], 

    ['жара', True]

)
dct_1
dct_1.setdefault('погода')
dct_1
dct_1.setdefault('погода', 'жара')
dct_1
dct_1.setdefault('погода на завтра', 'жара')
dct_1
dct_1.setdefault('погода на завтра', 'холод')
dct_1
dct_2 = dct_1.copy()

dct_2
dct_2.clear()
dct_2
dct_1
# Имя

# Фамилия

# Пол

# Дата рождения

# Возраст

# Итоговые оценки за месяц

# Итоговая контрольная



group = {

    '+79291111111':{

        'Имя':'Иван',

        'Фамилия':'Петров',

        'Пол':'м',

        'Дата рождения':'01-01-1991',

        'Возраст':23,

        'Итоговые оценки за месяц':(60, 65, 97, 40),

        'Итоговая контрольная':40.7,

    },

     '+79282222233':{

        'Имя':'Матвей',

        'Фамилия':'Васильев',

        'Пол':'м',

        'Дата рождения':'04-09-1996',

        'Возраст':20,

        'Итоговые оценки за месяц':(70, 75, 67, 40),

        'Итоговая контрольная':49.6,

    },

     '+79264443333':{

        'Имя':'Татьяна',

        'Фамилия':'Фоменко',

        'Пол':'ж',

        'Дата рождения':'01-01-1988',

        'Возраст':27,

        'Итоговые оценки за месяц':(20, 43, 87, 70),

        'Итоговая контрольная':70.2,

    }

}
group
group['+79264443333']
group.get('+79264443333')
for i in group.keys():

    print(i)

#     print(group['+79264443333'])
for i in group.keys():

     print(group[i])
for i in group.values():

    print(i)
group['+79264443333']
group['+79264443333']['Имя']
for i in group.keys():

     print(group[i]['Имя'])
for i in group.values():

    print(i)
for i in group.values():

    print(i['Имя'])
for i in group.values()['Имя']:

    print(i)
group['+79264443333']
group['+79264443333']['Итоговые оценки за месяц']
group['+79264443333']['Итоговые оценки за месяц'][-1]
group['+79264443333']['Итоговые оценки за месяц'][-1] = 20
lst = list(group['+79264443333']['Итоговые оценки за месяц'])
lst
lst[-1] = 20
lst
tpl = tuple(lst)
tpl
group['+79264443333']['Итоговые оценки за месяц']
group['+79264443333']['Итоговые оценки за месяц'] = tpl
group['+79264443333']['Итоговые оценки за месяц']
group
data = {

    'Имя':'Дарья',

    'Фамилия':'Игоревна',

    'Возраст':22,

}
group['+79298887766'] = data
group
data
group.update({'+79298887765':data})
group
group.setdefault('+79298887765', data)
del group['+79298887766']
group
group.pop('+79298887765')
group
month_1_lst = []

month_2_lst = []

month_3_lst = []

month_4_lst = []



for i in group.keys():

#     достучаться до значения итоговых оценок за месяц

    tpl = group[i]['Итоговые оценки за месяц']

    print(tpl)

    

#     Распаковываем эти значения

    month_1, month_2, month_3, month_4 = tpl    

    print(month_1)

    print(month_2)

    print(month_3)

    print(month_4)

    

#     Собирать значения для каждого месяца в отдельном списке

    month_1_lst.append(month_1)

    month_2_lst.append(month_2)

    month_3_lst.append(month_3)

    month_4_lst.append(month_4)
month_1_lst
month_1_lst, month_2_lst, month_3_lst, month_4_lst
print(

    sum(month_1_lst) / len(month_1_lst), 

    sum(month_2_lst) / len(month_2_lst),

    sum(month_3_lst) / len(month_3_lst), 

    sum(month_4_lst) / len(month_4_lst)

)
sum(month_1_lst) / len(month_1_lst), 

sum(month_2_lst) / len(month_2_lst),

sum(month_3_lst) / len(month_3_lst), sum(month_4_lst) / len(month_4_lst)
sum(month_1_lst) / len(month_1_lst), sum(month_2_lst) / len(month_2_lst), sum(month_3_lst) / len(month_3_lst), sum(month_4_lst) / len(month_4_lst)
sum(month_1_lst) / len(month_1_lst), \

sum(month_2_lst) / len(month_2_lst), \

sum(month_3_lst) / len(month_3_lst), \

sum(month_4_lst) / len(month_4_lst)