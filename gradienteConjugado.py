import sympy as sym
import numpy as np
# código para tres dimensiones
n = 2
iterMax = 10
x = np.zeros((n, iterMax + 1))
# punto inicial
x[0][0] = 0
x[1][0] = 3

alfa = np.zeros(iterMax + 1)

d = np.zeros((n, iterMax + 1))
i = 0
x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')
# función
f1 = (x1 - 2)**4 + (x1 -2*x2)**2
# derivada parciales
dfx1 = sym.diff(f1, x1)
dfx2 = sym.diff(f1, x2)
alf = sym.Symbol('alf')
epsilon = 0.000001
while True:
    # Evaluar f(x_i)
    sol = f1.subs([(x1, x[0][i]), (x2, x[1][i])])
    # Gradiente
    d[0][i] = -1 * dfx1.subs([(x1, x[0][i]), (x2, x[1][i])])
    d[1][i] = -1 * dfx2.subs([(x1, x[0][i]), (x2, x[1][i])])

    # reemplazar en f(x_i + alfa * d_i)
    f2 = f1.subs([(x1, x[0][i] + alf * d[0][i]), (x2, x[1][i] + alf * d[1][i])])

    # resolver f(x_i + alfa * d_i)
    sol1 = sym.solve(sym.diff(f2, alf))

    # imprimir
    print(i, [x[0][i], x[1][i]], sol, [d[0][i], d[1][i]], sol1[0])

    alfa[i] = sol1[0]

    # condiones para terminar
    if i == iterMax or abs(x[0][i] - x[0][i - 1]) <= epsilon and abs(x[1][i] - x[1][i - 1]) <= epsilon:
        break
    # recalcular x_{i + 1}
    x[0][i + 1] = x[0][i] + alfa[i] * d[0][i]
    x[1][i + 1] = x[1][i] + alfa[i] * d[1][i]

    i = i + 1
