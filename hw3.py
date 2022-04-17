"""
Luis Rangel DaCosta
16 Apr. 2022

Implementation of Central Path Algorithm for minimization of 1^T x s.t.
-1 <= x_i <= 1
"""

import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

class log_barrier():

    def __init__(self, constraints=None):
        self.constraints = constraints

    def grad(self, x):
        terms = [a[1]/(a[0]-a[1].T@ x) for a in self.constraints]
        return reduce(lambda x,y: x+y, terms)

    def hessian(self, x):
        terms = [(a[1] @ a[1].T)/((a[0]-a[1].T@ x)**2.0) for a in self.constraints]
        return reduce(lambda x,y: x+y, terms)

    def hessian_inv(self, x):
        return np.linalg.inv(self.hessian(x))

def local_norm(barrier, x, y):
    return np.sqrt(y.T @ barrier.hessian_inv(x) @ y)

def main():

    c = np.ones((20,1))
    constraints = [(1, x.T[:,None]) for x in np.eye(20)] \
                 + [(1,-x.T[:,None]) for x in np.eye(20)]

    F = log_barrier(constraints)

    t = 0
    x = np.zeros((20,1))
    gamma = 1

    x_s = [np.copy(x)]
    for i in range(1000):
        if local_norm(F, x, c) >  1e-16:
            t = t + gamma/local_norm(F, x, c)
        else:
            t = t + 1e-4
        p = t * c + F.grad(x)
        xi = local_norm(F, x, p)
        x = x - (1/(1+xi)) * F.hessian_inv(x) @ p
        x_s.append(np.copy(x))
    

    x_s = [np.linalg.norm(-1*np.ones((20,1))-y) for y in x_s]

    fig = plt.figure()
    ax = fig.gca()
    ax.loglog(x_s)
    ax.set_xlabel("N iterations")
    ax.set_ylabel("$||x-x^*||$")
    plt.show()

if __name__ == "__main__":
    main()