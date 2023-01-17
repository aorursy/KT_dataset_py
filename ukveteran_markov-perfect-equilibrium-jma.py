import numpy as np

import quantecon as qe

import matplotlib.pyplot as plt

%matplotlib inline
# Parameters

a0 = 10.0

a1 = 2.0

β = 0.96

γ = 12.0



# In LQ form

A = np.eye(3)

B1 = np.array([[0.], [1.], [0.]])

B2 = np.array([[0.], [0.], [1.]])





R1 = [[      0.,     -a0 / 2,          0.],

      [-a0 / 2.,          a1,     a1 / 2.],

      [       0,     a1 / 2.,          0.]]



R2 = [[     0.,           0.,      -a0 / 2],

      [     0.,           0.,      a1 / 2.],

      [-a0 / 2,      a1 / 2.,           a1]]



Q1 = Q2 = γ

S1 = S2 = W1 = W2 = M1 = M2 = 0.0



# Solve using QE's nnash function

F1, F2, P1, P2 = qe.nnash(A, B1, B2, R1, R2, Q1, 

                          Q2, S1, S2, W1, W2, M1, 

                          M2, beta=β)



# Display policies

print("Computed policies for firm 1 and firm 2:\n")

print(f"F1 = {F1}")

print(f"F2 = {F2}")

print("\n")
Λ1 = A - B2 @ F2

lq1 = qe.LQ(Q1, R1, Λ1, B1, beta=β)

P1_ih, F1_ih, d = lq1.stationary_values()

F1_ih
np.allclose(F1, F1_ih)
AF = A - B1 @ F1 - B2 @ F2

n = 20

x = np.empty((3, n))

x[:, 0] = 1, 1, 1

for t in range(n-1):

    x[:, t+1] = AF @ x[:, t]

q1 = x[1, :]

q2 = x[2, :]

q = q1 + q2       # Total output, MPE

p = a0 - a1 * q   # Price, MPE



fig, ax = plt.subplots(figsize=(9, 5.8))

ax.plot(q, 'b-', lw=2, alpha=0.75, label='total output')

ax.plot(p, 'g-', lw=2, alpha=0.75, label='price')

ax.set_title('Output and prices, duopoly MPE')

ax.legend(frameon=False)

plt.show()
# == Parameters == #

a0 = 10.0

a1 = 2.0

β = 0.96

γ = 12.0



# == In LQ form == #

A  = np.eye(3)

B1 = np.array([[0.], [1.], [0.]])

B2 = np.array([[0.], [0.], [1.]])

R1 = [[      0.,      -a0/2,          0.],

      [-a0 / 2.,         a1,     a1 / 2.],

      [       0,    a1 / 2.,          0.]]



R2 = [[     0.,          0.,     -a0 / 2],

      [     0.,          0.,     a1 / 2.],

      [-a0 / 2,     a1 / 2.,          a1]]



Q1 = Q2 = γ

S1 = S2 = W1 = W2 = M1 = M2 = 0.0



# == Solve using QE's nnash function == #

F1, F2, P1, P2 = qe.nnash(A, B1, B2, R1, R2, Q1,

                          Q2, S1, S2, W1, W2, M1,

                          M2, beta=β)
AF = A - B1 @ F1 - B2 @ F2

n = 20

x = np.empty((3, n))

x[:, 0] = 1, 1, 1

for t in range(n-1):

    x[:, t+1] = AF @ x[:, t]

q1 = x[1, :]

q2 = x[2, :]

q = q1 + q2       # Total output, MPE

p = a0 - a1 * q   # Price, MPE
R = a1

Q = γ

A = B = 1

lq_alt = qe.LQ(Q, R, A, B, beta=β)

P, F, d = lq_alt.stationary_values()

q_bar = a0 / (2.0 * a1)

qm = np.empty(n)

qm[0] = 2

x0 = qm[0] - q_bar

x = x0

for i in range(1, n):

    x = A * x - B * F * x

    qm[i] = float(x) + q_bar

pm = a0 - a1 * qm
fig, axes = plt.subplots(2, 1, figsize=(9, 9))



ax = axes[0]

ax.plot(qm, 'b-', lw=2, alpha=0.75, label='monopolist output')

ax.plot(q, 'g-', lw=2, alpha=0.75, label='MPE total output')

ax.set(ylabel="output", xlabel="time", ylim=(2, 4))

ax.legend(loc='upper left', frameon=0)



ax = axes[1]

ax.plot(pm, 'b-', lw=2, alpha=0.75, label='monopolist price')

ax.plot(p, 'g-', lw=2, alpha=0.75, label='MPE price')

ax.set(ylabel="price", xlabel="time")

ax.legend(loc='upper right', frameon=0)

plt.show()