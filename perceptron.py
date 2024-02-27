import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import abs, sin, cos, exp
from plotnine import ggplot, aes, geom_point, xlim, ylim, stat_function

np.random.seed(1234)


def f(x):
    return -cos(-x/2) + sin(4/(abs(x)+0.3))+0.2*x


def fermi(a, derivative=False):
    return 1/(1+exp(-a)) if not derivative else exp(-a)/(exp(-a)+1)**2


def linear(a, derivative=False):
    return a if not derivative else np.ones_like(a)


def visualize_f(points):
    g_df = pd.DataFrame(points, columns=["x", "y"])
    limits = (-10, 10)
    p = ggplot(g_df, aes(x="x", y="y")) +\
        geom_point() + xlim(limits) + ylim(limits) +\
        stat_function(fun=f, color="red")
    print(p)


def visualize_error(points):
    g_df = pd.DataFrame(points, columns=["x", "y"])
    xlimits = (-1, points.shape[0])
    ylimits = (0, points[0, 1]*1.1)
    fig, ax = plt.subplots()
    ax.scatter(g_df["x"], g_df["y"])
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_xlabel("epoch")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Mean squared Error over time")
    plt.show()


x = np.resize(np.linspace(-10, 10, 1001), (1001, 1))
y = np.resize(f(x), (1001, 1))
nodes = 20

w_hid = 2*np.random.random((x.shape[1] + 1, nodes)) - 1
w_out = 2*np.random.random((nodes + 1, y.shape[1])) - 1

n = 1000
learnrate = 0.01

g = []
error = []

for epoch in range(n):
    # hstack to add the bias
    o_in = np.hstack((np.ones((x.shape[0], 1)), x))
    a_hid = o_in @ w_hid
    o_hid = np.hstack((np.ones((x.shape[0], 1)), fermi(a_hid)))
    a_out = o_hid @ w_out
    o_out = linear(a_out)

    square_error = (o_out - y)**2
    average_square_error = np.sum(square_error)/square_error.shape[0]
    error.append(average_square_error)

    d_out = fermi(a_out, derivative=True) * (o_out - y)
    d_hid = linear(a_hid, derivative=True) * (w_out @ d_out.T).T[:, 1:]

    w_out += - learnrate * \
        np.mean(o_hid * d_out, axis=0, keepdims=True).T
    w_hid += - learnrate * \
        np.mean(d_hid[:, np.newaxis, :] * o_in[:, :,
                np.newaxis], axis=0, keepdims=False)

    if epoch == n-1:
        g = np.hstack((x, o_out))

visualize_f(g)
error = np.stack((np.arange(0, n), error), axis=1)
visualize_error(error)
