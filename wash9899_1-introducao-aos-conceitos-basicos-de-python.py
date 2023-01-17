# Você vai executar seu primeiro comando Python. Apenas execute com o botão ► e veja a saída.

print("Olá! Esta é a minha primeira execução em Python!")
# Adicione uma célula abaixo desta. Imprima uma frase inspiradora e execute trecho de código. Depois clique em Commit.

print("Esta é uma frase inspiradora :-D")
# Coloque seu nome dentro das aspas duplas (" ")

nome = "Washington"



# Poderia ser também em aspas simples. Coloque seu sobrenome.

sobrenome = 'Oliveira'



# Imprime

print('Meu nome é: ', nome, sobrenome)
# Imprime o Tipo da Variável

print(type(nome))
# Agora, e se o nome fosse John O' Brian, como imprime isso?

print("John O' Brian")
# Lembre que Python começa a contar do zero 0. O que a linha abaixo irá retornar? Pense antes de executar..

print(nome[0])
# Python também sabe contar de trás pra frente. Tente advinhar que a linha abaixo irá retornar.

print(nome[-1])
# E retorna partes de um index também, inclui o primeiro e não inclui o último do intervalo (range)

print(sobrenome[1:3])
nome.isnumeric()
# descubra quais são os métodos disponíveis para a variável sobrenome. Escolha o método upper e veja o resultado

sobrenome.upper()
# 1. Crie 2 variáveis: `moeda` com valor de "Dolar" e `cotacao` com o valor de 3.84. **Dica:** o valor em float é separado por ponto e não por vírgula. Exemplo: cotacao_anterior = 3.84   (e não 3,84)

moeda = 'Dolar'

cotacao = 3.85
# 2. Imprima o tipo de cada variável

print(type(moeda))

print(type(cotacao))
# 3. Calcule o valor de 3x a cotacao e coloque o resultado numa nova variável chamada `resultado`. Imprima o resultado.

resultado = 3 * cotacao

print('Resultado de 3 * %5.2f = %5.2f' % (cotacao,resultado))
# 1. Converta o resultado para o tipo int. O que aconteceu com as casas decimais?

print(int(resultado))

# As casas decimais foram truncadas
# 2. Converta o resultado para string e imprima o resultado

print(str(resultado))
# Recebe valores lógicos

a = True

b = False



# Compara valores. LEMBREM: Python é case sensitive

Not(a or b) 
# Compara valores

not(a or b)
# Qual o Resultado de 2+(2+10%3)*4 

print(2+(2+10%3)*4)

# E qual o Resultado de 2+2+10%3*4

print(2+2+10%3*4)
# Advinhe rápido, qual o resultado desta expressão? É True ou False? Pense antes de executar..

True or True and False
# Imprima o valor mínimo entre (3,6,1)

print(min(3,6,1))

# Imprima o valor absoluto de (-35)

print(abs(-35))
help(round)
def menor_diferenca(a, b, c):

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)
# Chama a funcao com valores diversos para ver o resultado

print(menor_diferenca(1,10,100))

print(menor_diferenca(1,10,10))
# Avalia a documentacao da funcao criada...

help(menor_diferenca)
# Ihh.. nao tem nenhuma documentacao. Vamos criar com docstring

def menor_diferenca(a, b, c):

    """Retorna a menor diferença absoluta entre 2 números entre a, b e c.

    >>> menor_diferenca(1, 5, -5)

    4

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)



# Agora tem alguma documentação

help(menor_diferenca)
def menor_diferenca(a, b, c):

    """Retorna a menor diferença absoluta entre 2 números entre a, b e c.

    >>> menor_diferenca(1, 5, -5)

    4

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    min(diff1, diff2, diff3)

    

print(menor_diferenca(1,10,100))

print(menor_diferenca(1,10,10))
def tinder(nome="Jhenifer"):

    print("O nome dela é", nome)

    

# Chama a funcao sem argumento: usa o argumento padrão

tinder()
# Chama a funcao passando argumento: sobrescreve o argumento padrao nesta chamada

tinder(nome="Eva")
# Crie a funcao

def formata_numero(numero=3.141592):

    """

    Retorna um número com 2 casas decimais.

    """

    return(str(round(numero,2)))
# Chama a funcao sem passar valor

formata_numero()
# Chama a funcao com o valor 5.13145

formata_numero(5.13145)
# Qual o erro desta linha abaixo? Resp: o nome da função round() está errado

ruound(3.14152, 2)
# Qual o erro deste código abaixo?

x = -10

y = 5



# Retorna o menor valor absoluto entre duas variáveis x e y

smallest_abs = min(abs(x,y))

# A função abs() só permite um argumento
# Qual o erro deste código abaixo?

def f(x):

    y = abs(x)

return y



print(f(5))

# O return está sem identação
# Segue um exemplo básico de condições:

def inspecao(x):

    if x == 0:

        print(x, "é zero")

    elif x > 0:

        print(x, "é positivo")

    elif x < 0:

        print(x, "é negativo")

    else:

        print(x, "é diferente de tudo que já vi...")



        

# Chama a função

inspecao(0)
inspecao(complex(3,4))
# Executa a função passando outro valor como parâmetro

inspecao(-15)
# Após ( : ) as linhas identadas pertencem ao corpo da função e a linha que não foi identada representa a finalização da função

def f(x):

    if x > 0:

        print("Imprime x se for positivo; x =", x)

        print("Também imprime x se for positivo; x =", x)

    print("Sempre imprime x, independente do seu valor; x =", x)



f(1)
# Qual será o resultado dessa execução? Imagine antes de executar. 

f(0)
def teste(nota):

    if nota < 50:

        resultado = 'reprovou'

    else:

        resultado = 'passou'

    print('Voce', resultado, 'no teste com a nota de', nota)

    

teste(80)
def teste(nota):

    resultado = 'reprovou' if nota < 50 else 'passou'

    print('Voce', resultado, 'no teste com a nota de', nota)

    

teste(45)
# Crie uma função que receba um numero como argumento e retorne `True` se o numero for positivo e retorne `False` se o número for negativo

def eh_positivo(numero):

    if numero > 0:

        return True

    elif numero < 0:

        return False

    else:

        return 'Valor 0'



print(eh_positivo(-4))
# Lista simples

primos = [2,3,5]

dias = ['Segunda', 'Terça', 'Quinta']



# O que a linha abaixo vai imprimir?

dias[2]
# E agora?

dias[0:2]
# Lista de lista

# Pode ser criada assim...

dias = [

    [2, 'Segunda'],

    [3, 'Terça'],

    [5, 'Quinta']  # Usar virgula depois do último elemento é opcional

]



# ... ou assim e terá o mesmo resultado que a anterior

dias = [[2, 'Segunda'],[3, 'Terça'],[5, 'Quinta']]



# O que a linha abaixo vai imprimir?

dias[2]
semana =  ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta']

print(semana)



# Aqui irá substituir o primeiro elemento da lista, contando a partir de zero (0)

semana[0] = 'Seg'

semana
# Crie uma lista que contenha os meses de Janeiro a Julho: meses

meses = ['janeiro','fevereiro','março','abril','maio','junho','julho']

# Depois imprima apenas o mês de Fevereiro.

meses[1]

# Por fim, substitua os meses de Janeiro e Fevereiro por Agosto e Setembro.

meses[0] = 'agosto'

meses[1] = 'setembro'

# Adicione um novo mês na lista, o mês de Outubro

meses.append('outubro')
meses
# Quantos meses tem nesta lista?

meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho']

len(meses)



# Ordena a lista na ordem alfabética

sorted(meses)
help(sorted)
# Exemplo de Loop For iterando dentro de uma lista

planetas = ['Marte', 'Mercúrio', 'Terra', 'Jupiter']

for planet in planetas:

    print(planet)
# Exemplo de Loop For iterando dentro de uma string

texto = 'Lei Especial de Gratificação ApLicada.'

msg = ''



# O que o código abaixo vai fazer?

for char in texto:

    if char.isupper():

        print(char, end='-')
# Exemplo de loop que repete 5 vezes

for i in range(5):

    print('Imprimindo o número i = ', i)
i = 0

while i < 10:

    print('Valor de i = ', i)

    i = i + 1
print('{:,.2f}'.format(4234234.329))
# Chame 5 pessoas com nomes de: Gates, Madog, 'Bob Esponja', 'Fabio Assunção' e 'Alan Turing' para a sua reunião

pessoas = ['Bill Gates','Mad Dog','Bob Esponja','Fabio Assunção','Alan Turing']

for pessoa in pessoas:

    print("Olá,",pessoa,", teremos uma reunião nesta segunda-feira às 10h")
# 1.1 RESPOSTA

# 1. Crie 2 variáveis: `moeda` com valor de "Dolar" e `cotacao` com o valor de 3.84. **Dica:** o valor em float é separado por ponto e não por vírgula. Exemplo: cotacao_anterior = 3.84   (e não 3,84)

moeda = "Dólar"

cotacao = 3.84



# 2. Imprima o tipo de cada variável moeda e cotação. Quais são seus tipos?

type(moeda)

type(cotacao)



# 3. Calcule o valor de 3x a cotacao e coloque o resultado numa nova variável chamada `resultado`. Imprima o resultado.

resultado = 3 * cotacao

resultado
# 1.2 RESPOSTA

# 1. Converta o resultado para o tipo int. O que aconteceu com as casas decimais?

int(resultado)



# 2. Converta o resultado para string e imprima o resultado

str(resultado)
# 2. RESPOSTA

# Qual o Resultado de 2+(2+10%3)*4 

print(2+(2+10%3)*4)



# Qual o Resultado de 2+2+10%3*4

print(2+2+10%3*4)
# 3.1 RESPOSTA

# Imprima o valor mínimo entre (3,6,1)

print(min(3,6,1))



# Imprima o valor absoluto de (-35)

print(abs(-35))
# 3.2 RESPOSTA

# Crie a funcao

def formata_numero(num=3.14159):

    """Recebe um numero com varias casas decimais como argumento e retorna o valor formatado para duas casas decimais.

    >>> formata_numero(3.14159)

    3.14

    """

    resultado = round(num,2)

    return(resultado)



# Chama a funcao sem passar valor

print(formata_numero())



# Chama a funcao com o valor 5.13145

print(formata_numero(num=5.13145))
# 4. RESPOSTA

# Crie uma função que receba um numero como argumento e retorne `True` se o numero for positivo e retorne `False` se o número for negativo

# Resposta longa

def positivo(numero):

    if numero > 0:

        return True

    else:

        return False



# Resposta curta

def positivo(numero):

    return numero > 0



# Testa a função

positivo(5)

positivo(-3)
# 5. RESPOSTA

# Crie uma lista que contenha os meses de Janeiro a Julho: meses

meses = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho']



# Depois imprima apenas o mês de Fevereiro.

meses[1]



# Por fim, substitua os meses de Janeiro e Fevereiro por Agosto e Setembro.

meses[:1] = ['Agosto', 'Setembro']



# Adicione um novo mês na lista, o mês de Outubro

meses[7] = 'Outubro'



# Imprima a lista de meses

meses
# 6. RESPOSTA

# Chame 5 pessoas com nomes de: Gates, Madog, 'Bob Esponja', 'Fabio Assunção' e 'Alan Turing' para a sua reunião

nomes = ['Gates', 'Madog', 'Bob Esponja', 'Fabio Assunção', 'Alan Turing']

for  i in nomes:

    print("Olá ", i , "teremos uma reunião nesta segunda-feira as 10h.")