#We have constructed a way to find the “best” b and m values using gradient descent!
import seaborn

import matplotlib.pyplot as plt
def get_gradient_at_b(x, y, b, m):

  N = len(x)

  diff = 0

  for i in range(N):

    x_val = x[i]

    y_val = y[i]

    diff += (y_val - ((m * x_val) + b))

  b_gradient = -(2/N) * diff  

  return b_gradient
def get_gradient_at_m(x, y, b, m):

  N = len(x)

  diff = 0

  for i in range(N):

      x_val = x[i]

      y_val = y[i]

      diff += x_val * (y_val - ((m * x_val) + b))

  m_gradient = -(2/N) * diff  

  return m_gradient
#step_gradient function here

def step_gradient(b_current, m_current, x, y,learning_rate):

    b_gradient = get_gradient_at_b(x, y, b_current, m_current)

    m_gradient = get_gradient_at_m(x, y, b_current, m_current)

    b = b_current - (learning_rate * b_gradient)

    m = m_current - (learning_rate * m_gradient)

    return [b, m]

  
# gradient_descent function here:  



months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

revenue = [52, 74, 79, 95, 115, 110, 129, 126, 147, 146, 190, 200]



def gradient_descent(x,y,learning_rate, num_iterations):

  b = 0

  m = 0

  for i in range(num_iterations):

    b,m=step_gradient(b,m,x,y,learning_rate)

  return b,m

b, m = gradient_descent(months, revenue, 0.01, 1000)
y = [m*x + b for x in months]



plt.plot(months, revenue, "o")

plt.plot(months, y)



plt.show()




