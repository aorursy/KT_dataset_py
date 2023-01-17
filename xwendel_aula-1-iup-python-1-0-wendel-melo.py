a = 1    #atribui o valor 1 à variável a
a       #ao digitar o nome de uma variável no prompt interativo 
b = a

b
b is a   # o operaor is retorna True se duas variáveis apontam para o mesmo objeto. Ele retornará False caso não apontem
a = 2
a
res = True

tas = False
resultado = 2 ** 1000   #o tipo int consegue manipular grandes inteiros, por exemplo 2 elevado a 1000

resultado
pi = 3.1435

w = 7.349
inteiro = 5  #dado do tipo int (pois não tem o ponto decimal)

real = 5.0   #dado do tipo float (tem o ponto decimal)
c = 1 + 3.2j    #numero complexo

d = 2.6 - 7.8j  #outro numero complexo
#Python suporta operações com números complexos naturalmente

somacomplexa = c + d

somacomplexa
c = None #a variável c será criada e apontará para um objeto None
p = 9

type(p)  #retorna um objeto type que representa o tipo do objeto apontado pela variável p
nome = "jessica"

nome[0]    #Podemos acessar um caractere da string pela sua posição, comoçando a parti do zero. O caracter na posição 0 é "j"
#todavia, como a string é um objeto imutável, não podemos trocar um determinado caractere por outro:

nome[j] = "G"   #Erro, pois a string é imutável
#observe a diferença de interpretação ao escrever um nome entre aspas e sem as aspas

resultado = 19

n = resultado  #como o nome resultado está sem aspas, Python interpreta que vc se refere ao conteúdo da variável resultado. Assim. n tb apontará p/ 19

n
m = "resultado" #aqui, resultado está entre aspas. 

#Então, Python interpreta que você quer realmente trabalhar com o texto "resultado" (literalmente), e não com a variável resultado

m
v = (7,  2.5,  'Carol',  None,  1)  #pode-se misturar tipos na mesma tupla

v
#pode-se acessar os alementos da tupla através de índices, começando a contar do zero.

v[2]
#no entanto, não se pode trocar o objeto em um determinado índice, pois a tupla éum objeto imutável

v[1] = 4    #erro, pois a tupla é imutável