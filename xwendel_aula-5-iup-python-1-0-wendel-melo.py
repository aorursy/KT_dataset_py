#declaramos uma função com a cláusula def. Em seguida, damos um nome para função e listamos seus argumentos de entrada

#vamos fazer uma função que recebe dois valores a e b e os soma



def soma(a, b):

    res = a + b

    return res
#para usar a função, basta fazer a chamada e passar argumentos

r = soma(2,3)

print(r)
#A variável res foi declarada dentro do contexto (escopo) da função soma. Ela não pode ser acessada de fora da função

print(res)  #erro
soma(7,4)
#note que a função deve receber dois argumentos. Se apenas um for passado, teremos um erro

soma(9)    #erro. Falta o argumento b
#podemos definir um valor default para um argumento de uma função. Assim, se o argumento não for passado, o valor default será assumido

def soma(a, b = 0):    #caso b não seja passado, será assumido que ele vale zero

    res = a + b

    return res
#agora podemos chamar a função soma passando apenas um único argumento

soma(9)
#por exemplo, podemos usar a função soma para somar duas strings

r = soma("jessica", "gomes") 

print(r)
#ou duas tuplas

resultado = soma( (1,3,5),  (2,4,6) )

print(resultado)
#pode usar o type para testar o tipo dos argumentos recebidos. Mas essa não é uma prática muito recomendada. 

#No geral, deseja-se deixar que uma função trabalhe com qualquer tipo de dado suportado pelas operações realizadas.

#Todavia, se desejarmos restringir a função soma apenas para inteiros:

def soma(a, b):

    if type(a) == int and type(b) == int:

        return a + b

    else:

        raise TypeError("Argumentos devem ser inteiros")        #levanta uma exceção (erro) do tipo TypeError
soma("wendel", "melo")
def fatorial(n):

    rfat = 1

    for k in range(n, 1, -1):

        rfat = rfat * k

    return rfat



if __name__ == "__main__":  #esse if testa se o programa está sendo executado como programa principal

    

    numero = int( input("Entre com um numero: ") )

    f = fatorial(numero)



    print("Fatorial de %s: %s"%(numero, f) )