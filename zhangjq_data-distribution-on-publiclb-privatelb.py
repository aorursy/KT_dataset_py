from sympy import *

x, y, z = symbols('x y z')
solution_public_LB = solve([(1/15*x)/(1/15*x+1/5*y+1/16*z)-0.28407,(1/5*y)/(1/15*x+1/5*y+1/16*z)-0.04916,x+y+z-1],[x,y,z])
print('Data distribution on public leaderboard')
print({'agreed':solution_public_LB.get(x),'disagreed':solution_public_LB.get(y),'unrelated':solution_public_LB.get(z)})

solution_private_LB = solve([(1/15*x)/(1/15*x+1/5*y+1/16*z)-0.35476,(1/5*y)/(1/15*x+1/5*y+1/16*z)-0.06680,x+y+z-1],[x,y,z])
print('Data distribution on private leaderboard')
print({'agreed':solution_private_LB.get(x),'disagreed':solution_private_LB.get(y),'unrelated':solution_private_LB.get(z)})