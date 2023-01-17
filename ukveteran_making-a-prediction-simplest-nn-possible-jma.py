weight=0.1

def neural_network(input, weight):

    prediction=input*weight

    return prediction
numberoftoes=[8.5,10,9,25]

input=numberoftoes[0]

pred=neural_network(input, weight)

print(pred)