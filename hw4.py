"""
Luis Rangel DaCosta
13 May 2022

Implementation of Damped Newton's method for 
f(x) = \frac{1}{\epsilon} \sum_{i=1}^{10} i x_i - \sum_{i=1}^{10} log(1-x_i^2)
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

class log_quadratic_with_linear():

    def __init__(self, eps, N=10):
        self.N = 10
        self.eps = eps

    def grad(self, x):
        # gradient terms do not mix
        linear_terms = (1/self.eps) * np.arange(1,11)[:,None] 
        log_terms = 2 * x / (1 - x**2.0)
        return linear_terms + log_terms

    def hessian(self, x):
        x2 = x**2.0
        x_pp = 2*(x2 + 1)/((1-x2)**2.0)
        return np.diag(x_pp[:,0])

    def hessian_inv(self, x):
        return np.linalg.inv(self.hessian(x))

def local_norm(f, x, y):
    return np.sqrt(f.grad(y).T @ f.hessian_inv(x) @ f.grad(y))

def main():
    x_0 = np.zeros([10,1])
    eps = 1
    M_f = 2

    fig = plt.figure()
    ax = fig.gca()
    for eps in [1, 0.1, 0.01, 0.005]:
        x = np.copy(x_0)
        F = log_quadratic_with_linear(eps=eps)
        llambda_list = []
        llambda=  local_norm(F, x, x)[0]
        while(llambda > 1e-6):
            xi = M_f * llambda
            H =  F.hessian_inv(x)
            G = F.grad(x)
            x = x - (1/(1+xi)) * H @ G
            llambda =  local_norm(F, x, x)[0]
            llambda_list.append(llambda)
            

        print("%i steps for eps=%0.4f" % (len(llambda_list), eps))
        print(x.T)
        ax.loglog(llambda_list, label="$\epsilon=%0.4f$" % eps)
    
    ax.axhline(y=0.25, color="k", linestyle="--", linewidth=0.5)
    ax.text(s="$\lambda_k(x_k) = 1/4$", y=0.3, x = 1)
    ax.legend()
    ax.grid()
    ax.set_xlabel("Iteration step")
    ax.set_ylabel("$\lambda_k(x_k)$")
    plt.show()  

if __name__ == "__main__":
    main()