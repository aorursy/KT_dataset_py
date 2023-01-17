#Precisamos fazer o download do ambiente de Connect X:

# Vamos precisar de um kaggle-enviroments customizado para a avaliação.

!pip install git+https://github.com/matheusgmaia/kaggle-environments

    
#Criar ambiente

from kaggle_environments import evaluate, make, utils



env = make("connectx", debug=True)

env.render()
# Exemplo de agente. Esse agente escolhe de maneira aleatória uma coluna que não esteja completa

import random

def my_agent(obs, cfg): #recebe o estado atual do jogo e a configuração do jogo

    coluna = random.choice([c for c in range(cfg.columns) if obs.board[c] == 0])

    return coluna
env.reset()

env.run([my_agent, "random"]) #Agente definido em my_agent versus angente randômico.

env.render(mode="ipython", width=500, height=450)
# Play as first position against random agent.

trainer = env.train([None, "random"])



observation = trainer.reset()



while not env.done:

    my_action = my_agent(observation, env.configuration)

    print("Ação do seu agente: Coluna", my_action+1)

    observation, reward, done, info = trainer.step(my_action)

    env.render(mode="ipython", width=100, height=90, header=False, controls=False)

env.render()
# "None" represents which agent you'll manually play as (first or second player).

env.play([None, my_agent], width=500, height=450) #Altere "rules" por my_agent para jogar contra o seu agente
#Sua vez

def my_agent(obs, cfg):

    '''

    Agente MiniMax:

    O agente implementado recebe o estado atual do jogo e prevê

    a jogada do adversário testando todas as possibilidades. O agente

    prevê, também, que o adversário é um agente inteligente então

    ele prevê que o adversário está tentando minimizar as possibilidades

    de vencer, logo, o agente aqui implementado fará a jogada que maximize

    as suas possibilidades de vitória.

    '''

    #print(obs)

    #print(cfg)

    return 0
def mean_win_draw(rewards):

    return sum( 1 for r in rewards if (r[0] == 1 or r[0] == 0.)) / len(rewards)



# Run multiple episodes to estimate its performance.

vs_random = mean_win_draw(evaluate("connectx", [my_agent, "random"], num_episodes=10))

print("My Agent vs Random Agent:", vs_random)



vs_negamax = mean_win_draw(evaluate("connectx", [my_agent, "negamax"], num_episodes=10))

print("My Agent vs Negamax Agent:", vs_negamax)



vs_rules = mean_win_draw(evaluate("connectx", [my_agent, "rules"], num_episodes=10))

print("My Agent vs Rule Agent:", vs_rules)



vs_greedy = mean_win_draw(evaluate("connectx", [my_agent, "greedy"], num_episodes=10))

print("My Agent vs Greedy Agent:", vs_greedy)
import csv



seu_nome = "SEU_NOME"



rows = [['Id', 'Predicted'],['random',vs_random],[ 'negamax', vs_negamax],[ 'rules', vs_rules],[ 'greedy', vs_greedy]]

f = open(seu_nome+'-ConnectX.csv', 'w')

with f:

    writer = csv.writer(f)

    for row in rows:

        writer.writerow(row)
import inspect

import os



def write_agent_to_file(function, file):

    with open(file, "a" if os.path.exists(file) else "w") as f:

        f.write(inspect.getsource(function))

        print(function, "written to", file)



write_agent_to_file(my_agent, "submission.py")
# Note: Stdout replacement is a temporary workaround.

import sys

out = sys.stdout

submission = utils.read_file("/kaggle/working/submission.py")

agent = utils.get_last_callable(submission)

sys.stdout = out



env = make("connectx", debug=True)

env.run([agent, agent])

print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")