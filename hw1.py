"""
Luis Rangel DaCosta
2/18/2022

EE 227C HW1

"""

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

class maxquad():

    def __init__(self, m, n):
        """

        Args:
            m: square size of matrices A_k 
            n: number of quadratic functions k x_T A_k x - <b_k, x>

        """
        self.m = m
        self.n = n
        iv = np.array([x for x in np.arange(1,self.m+1)])
        ii, jj = np.meshgrid(iv, iv)
        self.A = [np.exp(ii/jj)*np.cos(ii*jj)*np.sin(k) for k in range(1, self.n+1)]
        self.A = [A-np.tril(A)+np.triu(A).T for A in self.A]
        self.b = [(np.exp(iv/k)*np.sin(iv*k))[:,None] for k in range(1, self.n+1)]

        # I don't want to figure out the fastest way to set this for large m
        for k in range(1, self.n+1):
            for i in range(1, self.m):
                self.A[k-1][i-1, i-1] = (i/10)*np.abs(np.sin(k))
                for j in range(1, self.m):
                    if j != i:
                        self.A[k-1][i-1, i-1] += np.abs(self.A[k-1][i-1, j-1])

    def evaluate(self, x):
        return np.max([x.T @ A @ x - b.T @ x for A, b in zip(self.A, self.b)])

    def subgradient(self, x):
        """
        Return gradient of f_k(x) for the k that maximizes f(x) as subgradient.
        """
        k = np.argmax([x.T @ A @ x - b.T @ x for A, b in zip(self.A, self.b)])

        return 2*self.A[k] @ x - self.b[k]

def main():
    m = 10
    n = 5

    # fix the function, initial point, time horizons, and relevant constants
    f_maxquad = maxquad(m,n)
    x_1 = np.ones((m,1))
    horizons = [10**x for x in range(2,6)]
    C = 1

    # perform subgradient descent with fixed step size
    histories = {}
    for t in horizons:
        gamma_t = C/np.sqrt(t) 
        x = np.copy(x_1)
        fx = f_maxquad.evaluate(x)
        fx_l = [fx]
        min_fx = np.copy(fx)
        for i in range(2, t+1):
            g = f_maxquad.subgradient(x)
            x = x - gamma_t * (g/np.linalg.norm(g))
            fx = f_maxquad.evaluate(x)
            if fx < min_fx:
                min_fx = fx
            fx_l.append(min_fx)
        
        histories[t] = deepcopy(fx_l)

    f_star = np.min(histories[horizons[-1]])
    # perform subgradient with polyak step size
    histories_polyak = {}
    for t in horizons[-1:]:
        x = np.copy(x_1)
        fx = f_maxquad.evaluate(x)
        fx_l = [fx]
        min_fx = np.copy(fx)
        for i in range(2, t+1):
            g = f_maxquad.subgradient(x)
            g_norm = np.linalg.norm(g)
            gamma_t = (fx - f_star)/g_norm
            x = x - gamma_t * (g/g_norm)
            fx = f_maxquad.evaluate(x)
            if fx < min_fx:
                min_fx = fx
            fx_l.append(min_fx)
        
        histories_polyak[t] = deepcopy(fx_l)

    fig, axes = plt.subplots(1,2,figsize=(10,5))

    for t in horizons:
        axes[0].loglog(histories[t]-f_star, label=f"T={t}")
        
    axes[1].loglog(histories_polyak[horizons[-1]]-f_star)

    for ax in axes:
        ax.set_xlabel("Iteration")
        ax.set_ylim([1e-15, 1e5])
    
    axes[0].set_ylabel("$f(x)-f^*$")
    axes[0].set_title("Subgradient Descent with $C=1$ and $\gamma = \\frac{C}{\\sqrt{T}}$", fontsize=12)
    axes[0].legend()
    axes[1].set_title("Subgradient Descent with Polyak $\gamma_t = \\frac{f(x_t)-f^*}{||g(x_t)||}$", fontsize=12)
    
    plt.show()

    fig = plt.figure(figsize=(6,6))

    ax = fig.gca()
    for t in horizons:
        ax.loglog(histories[t]-f_star, label=f"T={t}")
        
    ax.loglog(histories_polyak[horizons[-1]]-f_star, label=f"Polyak")

    ax.set_xlabel("Iteration")
    ax.set_ylim([1e-15, 1e5])
    
    ax.set_ylabel("$f(x)-f^*$")
    ax.set_title("Subgradient Descent with $C=1$ and $\gamma = \\frac{C}{\\sqrt{T}}$ and Polyak $\gamma_t$", fontsize=12)
    ax.legend(loc="lower left")
    plt.show()



if __name__ == "__main__":

    main()