print('Меня зовут Андрей')
name = 'Игорь'



print('Меня зовут %s' % name)
surname = 'Петров'



print('Меня зовут %s. Моя фамилия %s' % (name, surname))
surname = 'Петров'



print('Меня зовут %s. Моя фамилия %s' % (name, name, name))
dct = {'name':name, 'surname':surname}



print('Меня зовут %(name)s Моя фамилия %(surname)s' % dct)
print('Моя фамилия %(surname)s. Меня зовут %(name)s' % dct)
from math import pi



pi
print('Число Пи равно = %f' % pi)
print('Число Пи равно = %.3f' % pi)
print('Число Пи равно = %10.3f' % pi)
print('Число Пи равно = %010.3f' % pi)
print('Число Пи равно = %10.3e' % pi)
print('Число Пи равно = %10.3e' % (pi/10000))
print('Число Пи равно = %010.3i' % 4.8)
print('Число Пи равно = %010.0f' % 4.8)
name, surname



print('Меня зовут %s Моя фамилия %s' % (name, surname))
name, surname



print('Меня зовут {} Моя фамилия {}'.format(name, surname))
print('Меня зовут {1} Моя фамилия {0}'.format(name, surname))
print('Меня зовут {1} Моя фамилия {1}'.format(name, surname))
dct = {'имя':name, 'фамилия':surname}



print('Меня зовут {имя} Моя фамилия {фамилия}'.format(**dct))
group_dict = {

    '+79999999999':{

        'name':'Василий',

        'surname':'Иванов',

        'age':23,

        'Score': 60.7

    },

    '+79999998888':{

        'name':'Василий',

        'surname':'Иванов',

        'age':20,

        'Score': 67.0

    },

    '+79999997777':{

        'name':'Анна',

        'surname':'Петрова',

        'age':27,

        'Score': 90.2

    },

}
print('Число равно {:.2f}'.format(pi))
print('Число равно {:+.3f}'.format(pi))
print('Число равно {:.0f}'.format(pi))
print('Число равно {:-.1f}'.format(-321))
print(f'Число равно {pi:.0f}')