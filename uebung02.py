import numpy as np
import pandas as pd
from numpy import abs, sin, cos, exp
from plotnine import ggplot, aes, geom_point, xlim, ylim, stat_function

np.random.seed(1001)


def f(x):
    return -cos(-x/2) + sin(4/(abs(x)+0.3))+0.2*x


def fermi(a):
    return 1/(1+exp(-a))


def fermi_d1(a):
    return (exp(-a)/(exp(-a)+1)**2)


def backpropagation(y, a):
    d = fermi_d1(a)*y
    return d


def perzeptron(input, weights):
    hidden_weights = weights[0]
    a_j = input @ hidden_weights
    o_j = fermi(a_j)
    outer_weights = weights[1]
    a_k = o_j @ outer_weights
    o_k = a_k
    return (a_j, o_j, a_k, o_k)


def train(input, expected, n, learnrate, units):
    hidden_weights = np.array(
        [np.random.rand(units)*2-1, np.random.rand(units)*2-1])
    outer_weights = np.random.rand(units)*2-1
    size = len(input)
    input = np.column_stack((np.ones(size), input))  # adding bias
    g = list()

    for epoch in range(0, n+1):
        average_mistake = 0
        for x, t in zip(input, expected):
            a_j, o_j, a_k, o_k = perzeptron(x, (hidden_weights, outer_weights))
            mistake = (o_k - t)**2
            average_mistake += mistake/size
            d_k = backpropagation(2*(o_k-t), a_k)
            d_j = backpropagation(np.sum(outer_weights * d_k), a_j)
            outer_weights += -learnrate*o_j*d_k
            hidden_weights += -learnrate * \
                (np.resize(x, (2, 1)) @ np.resize(d_j, (1, units)))
            if epoch == n:
                g.append((x[1], o_k))
        print(f"average mistake in epoch {epoch}: {average_mistake}")
    return g


def visualize(points):
    g_df = pd.DataFrame(points, columns=["x", "y"])
    limits = (-10, 10)
    p = ggplot(g_df, aes(x="x", y="y")) +\
        geom_point() + xlim(limits) + ylim(limits) +\
        stat_function(fun=f, color="red")

    print(p)


x = np.linspace(-10, 10, 1001)
y = f(x)

g = train(x, y, 100, 1, 20)
visualize(g)
