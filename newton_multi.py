import sympy as sym
import numpy as np

# código para tres dimensiones
n = 2
iterMax = 10
x=np.zeros((2,  iterMax + 1))
# punto inicial
x[0][0] = 0
x[1][0] = 3

alfa = np.zeros(iterMax + 1)
grad = np.zeros((n, iterMax + 1))
p = np.zeros(iterMax + 1)

i=0
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
f1 = (x1 - 2)**4 + (x1 -2*x2)**2
dfx1 = sym.diff(f1,x1)
dfx2 = sym.diff(f1,x2)

epsilon = 0.01
while True:

    sol=f1.subs([(x1, x[0][i]), (x2, x[1][i])])

    grad[0][i] = dfx1.subs([(x1, x[0][i]), (x2, x[1][i])])
    grad[1][i] = dfx2.subs([(x1, x[0][i]), (x2, x[1][i])])

    # calcular la hessiana
    T = sym.hessian(f1, (x1, x2))
    T = T.subs([(x1, x[0][i]), (x2,x [1][i])])
    T1 = np.matrix(T, dtype='float')
    T = np.linalg.inv(T1)

    # imprimir
    print(i, x[0][i], x[1][i], sol, grad[0][i], grad[1][i])

    # condiones para terminar
    if i == iterMax or abs(x[0][i] - x[0][i - 1]) <= epsilon and abs(x[1][i] - x[1][i - 1]) <= epsilon:
        break

    x[0][i + 1] = x[0][i] - np.dot(T[0,:], grad[:,i])
    x[1][i + 1] = x[1][i] - np.dot(T[1,:], grad[:,i])

    i += 1
