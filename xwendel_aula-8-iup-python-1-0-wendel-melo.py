d1 = { "nome":"jessica", "idade":29, "altura": 1.75 }

print(d1)
d2 = { 7: 3.14,   'pi': 3.14 }    #o mesmo elemento pode aparecer mais de uma vez em um diiconário, mas cada chave só pode aparecer uma vez
d3 = { 2:5,  9:{0:7, -1:10} }    #exemplo de dicionário dentro de dicionário

print(d3)
print(d1)
d1[0]    #ERRO! não faz sentido acessar elementos de dicionário através de índices
#os elementos de um dicionário só podem ser acessados através de suas respectivas chaves

d1["nome"]
#assim, podemos ter um dicionário onde usamos numeros inteiros como chave

duni = { 1:"UFU",   2: "UFRJ",  5:"UMICH" }

print(duni)
#assim, o acesso será feito com chaves inteiras, simulando, de certa forma, índices numéricos:

duni[2]
#para acrescentar um elemento, basta realizar uma atribuição sobre uma nova chave:

duni[4] = "FIOCRUZ"

print(duni)
#podemos usar o operador del para remover um elemento de um dicionario:

del duni[5]

print(duni)
#podemos alterar o elemento armazenado sobre uma chave:

duni[1] = 'Harvard'   #altera o elemento armazenao na chave 1 (Não se pode armazenar mais de um elemento na mesma chave)

print(duni)
#o operador len retorna o número de elementos em um dicionário

len(duni)
#os operadores in/not int verificam se um objeto consta como CHAVE em um diconário (cuidado p/não confundir)

2 in duni
"UFRJ" in duni  #O operador in NÃO verifica se um objeto é elemento de um dicionário
#podemos ver toda a listagem de métodos usando help

help(dict)
print(d1)
print( d1.keys() )

print( d1 )
print( d1.values() )
for chave in d1.keys():    #a variável chave percorrerá as chaves de d1 (em alguma ordem arbitrária)

    print( d1[chave] )
for elemento in d1.values():   #a variável elemento percorrerá os elementos de d1 diretamente (em alguma ordem arbitrária)

    print(elemento)
#percorrer "diretamente" um dicionário equivale a percorrer suas chaves (pode gera confusão para quem lê o código)

for v in d1:   #pode-se pensar que v percorrerá os elementos de d1, mas, na realidade, ele percorrerá suas chaves

    print(v)
#Função que lê uma string (texto) do usuário e retorna um dicionário com a frequência dos caracteres.

#Caracteres minusculos e maiusculos devem ser considerados equivalentes.



#no dicionário retornado, cada caracter da string será chave, e sua frequência, o respectivo elemento



def contaCaracteres(texto):

    textoMin = texto.lower()

    

    freqs = {}   #dicionario vazio para contar as frequências.

    

    for c in textoMin:     #percorre uma versão do texto só com caracteres minúsculos

        if c not in freqs:

            freqs[c] = 0

        

        freqs[c] += 1

    

    return freqs

    
contaCaracteres("Ananda")
contaCaracteres("O rato roeu a roupa do rei de roma")
#exemplo de uso de dicionários para manipulação arquivos JSON:

#programa que 



import requests



cep = 38408100



resposta = requests.get( "https://viacep.com.br/ws/%s/json/"%cep )

if resposta.status_code != 200:

    print("Erro no acesso à API viacep!")

else:

    dados = resposta.json()

    print(dados)


