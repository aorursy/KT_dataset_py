import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import torch
CUDA = True
N = 15000
np.random.seed(123456) # For reproducibility
seed_out = torch.manual_seed(123456)
dtype = torch.FloatTensor

if CUDA:
    torch.cuda.manual_seed(123456)
    dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

#First, build the "true" dataset with N=50 datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
x = torch.linspace(-5, 5, N).type(dtype)
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties
#   and non-trivial correlated errors.
yerr = 0.1 + 0.4 * torch.rand(N).type(dtype)
yerr_hom = 0.4*torch.ones(N).type(dtype)
hom_cov = torch.diag(yerr_hom ** 2).type(dtype)
iid_cov = torch.diag(yerr ** 2).type(dtype)
true_cov = 0.5 * torch.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + torch.diag(yerr ** 2)
y_noised = torch.distributions.multivariate_normal.MultivariateNormal(y, covariance_matrix=true_cov).sample()
y_reshaped = y_noised.type(dtype).view(N, 1)
y_reshaped.shape, type(y_reshaped)
A = torch.ones((N,2)).type(dtype)
A[:, 0] = x

AT= A.t()
C = true_cov

t0 = time.time()
factor = torch.cholesky(C)
S_inv = torch.mm(AT, torch.potrs(A, factor))
S = torch.inverse(S_inv) #only a 2 x 2, so it's cheap.
part1 = torch.potrs(y_reshaped, factor)
part2 = torch.mm(AT, part1)
ls_m_CU, ls_b_CU = torch.mm(S, part2)
t1 = time.time()

net_time = t1-t0
print(" m: {:.2f} \n b: {:.2f} \n time: {}".format(ls_m_CU[0], ls_b_CU[0], net_time))

#First, build the "true" dataset with N datapoints from a line model y=mx+b.
true_m, true_b = 0.5, -0.25
x = np.linspace(-5, 5, N)
y = true_m * x + true_b

#Introduce some noise with both measurement uncertainties
#   and non-trivial correlated errors.
yerr = 0.1 + 0.4 * np.random.rand(N)
yerr_hom = 0.4*np.ones(N)
hom_cov = np.diag(yerr_hom ** 2)
iid_cov = np.diag(yerr ** 2)
true_cov = 0.5 * np.exp(-0.5 * (x[:, None]-x[None, :])**2 / 1.3**2) + np.diag(yerr ** 2)
#y_noised = np.random.multivariate_normal(y, true_cov)
y_noised_cpu = y_noised.cpu().numpy()
coeffs = np.polyfit(x, y_noised_cpu, 1, w=1/yerr)
y_fit_func = np.poly1d(coeffs)
y_fit = y_fit_func(x)
plt.plot(x, y_noised_cpu, 'ko', alpha=0.5, label='Noisy, correlated data')
plt.plot(x, y, 'r-', alpha=1.0, label='$y = mx +b$', lw=3)
plt.plot(x, y_fit, 'g-', alpha=1.0, label='$\hat y = \hat m x + \hat b$', lw=3)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best');
#Linear algebra
A = np.vander(x, 2)
AT= A.T
C = iid_cov

t0 = time.time()
factor = cho_factor(C, overwrite_a=True)
S_inv = np.dot(AT, cho_solve(factor, A))
S = np.linalg.inv(S_inv)
ls_m, ls_b = np.dot(S, np.dot(AT, cho_solve(factor, y_noised_cpu)))
t1 = time.time()

net_time = t1-t0
print(" m: {:.2f} \n b: {:.2f} \n time: {}".format(ls_m, ls_b, net_time))
plt.plot(x, y_noised_cpu, 'ko', alpha=0.5, label='Noisy, correlated data')
plt.plot(x, y, 'r-', alpha=1.0, label='$y = mx +b$', lw=3)
plt.plot(x, ls_m*x+ls_b, 'b-', alpha=1.0, label='$\hat y = \hat m x + \hat b$ with C', lw=3)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best');
