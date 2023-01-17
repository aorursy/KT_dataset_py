import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Set seed and generate a_t sequence

np.random.seed(123)

n = 100

a_seq = np.sin(np.linspace(0, 5 * np.pi, n)) + 2 + 0.1 * np.random.randn(n)



def plot_simulation(γ=0.8, m=1, h=1, y_m=2):



    d = γ * np.asarray([1, -1])

    y_m = np.asarray(y_m).reshape(m, 1)



    testlq = LQFilter(d, h, y_m)

    y_hist, L, U, y = testlq.optimal_y(a_seq)

    y = y[::-1]  # Reverse y



    # Plot simulation results



    fig, ax = plt.subplots(figsize=(10, 6))

    p_args = {'lw' : 2, 'alpha' : 0.6}

    time = range(len(y))

    ax.plot(time, a_seq / h, 'k-o', ms=4, lw=2, alpha=0.6, label='$a_t$')

    ax.plot(time, y, 'b-o', ms=4, lw=2, alpha=0.6, label='$y_t$')

    ax.set(title=rf'Dynamics with $\gamma = {γ}$',

           xlabel='Time',

           xlim=(0, max(time))

          )

    ax.legend()

    ax.grid()

    plt.show()



plot_simulation()
plot_simulation(γ=5)
plot_simulation(γ=10)