import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def euclidean(c, x):
    return cdist(c, x)


class NN:
    def __init__(self, lr=0.1, k=50, bias=False):
        self.lr = lr
        self.bias = bias
        self.o_hid = 0
        self.o_out = 0
        self.k = k
        self.sigma = None
        self.weights_hid = None
        self.weights_out = np.random.uniform(-1, 1, (k + 1 * bias, 2))

    def fit(self, x):
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(x)
        self.weights_hid = kmeans.cluster_centers_

        distances = euclidean(x, self.weights_hid)
        self.sigma = distances.max() / np.sqrt(2 * self.k)  # calculate an appropriate sigma
        self.weights_hid = np.hstack(
            (self.weights_hid, np.ones((self.weights_hid.shape[0], 1 * self.bias))))  # add bias

    def forward(self, x):
        o_in = np.hstack((x, np.ones((x.shape[0], 1 * self.bias))))  # add bias if activated
        a_hid = euclidean(o_in, self.weights_hid)
        self.o_hid = np.hstack((self.gaussian(a_hid), np.ones((a_hid.shape[0], 1 * self.bias))))
        self.o_out = self.o_hid @ self.weights_out
        return self.o_out

    def backward(self, expected):
        self.weights_out += self.lr * (((expected - self.o_out).T @ self.o_hid) / 500).T
        # print(np.full((2, 500), 1 / 500) @ ((expected - self.o_out) ** 2))

    def gaussian(self, x, mu=0):
        return np.exp(-np.power(x - mu, 2) / (2 * np.power(self.sigma, 2))) / (self.sigma * np.sqrt(2 * np.pi))

    def visualize_hidden_units(self):
        fix, ax = plt.subplots()
        x_1 = np.squeeze(self.weights_hid[:, 0])
        x_2 = np.squeeze(self.weights_hid[:, 1])
        ax.scatter(x_1, x_2, color="green", label="hidden Layer Neurons")
        ax.set_xlabel("x_1")
        ax.set_ylabel("x_2")
        ax.legend()
        plt.show()


def visualize_datapoints(points, y):
    a = points[y[:, 0] > y[:, 1]].T
    b = points[y[:, 0] < y[:, 1]].T
    c = points[y[:, 0] == y[:, 1]].T

    fix, ax = plt.subplots()
    ax.scatter(a[0], a[1], color="red", label="A")
    ax.scatter(b[0], b[1], color="blue", label="B")
    ax.scatter(c[0], c[1], color="grey", label="C")
    ax.set_xlim(np.min(np.hstack((a[0], b[0], c[0]))) *
                1.1, np.max(np.hstack((a[0], b[0], c[0]))) * 1.1)
    ax.set_ylim(np.min(np.hstack((a[1], b[1], c[1]))) *
                1.1, np.max(np.hstack((a[1], b[1], c[1]))) * 1.1)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(1234)
    u = np.arange(1, 251)
    x1A = 1 + np.sin(0.1 * u + 1.5) * np.sqrt((u + 10))
    x2A = -2 + np.cos(0.1 * u + 1.5) * np.sqrt((u + 10))
    yA = np.vstack((np.ones((250, 1)), np.zeros((250, 1))))

    x1B = 1 + np.sin(0.1 * u - 1.6) * np.sqrt(u + 10)
    x2B = -2 + np.cos(0.1 * u - 1.6) * np.sqrt(u + 10)
    yB = np.vstack((np.zeros((250, 1)), np.ones((250, 1))))

    x1 = np.hstack((x1A, x1B))
    x2 = np.hstack((x2A, x2B))
    y = np.hstack((yA, yB))
    points = np.vstack((x1, x2)).T
    model = NN(k=50, lr=0.01, bias=False)
    model.fit(points)
    output = 0

    visualize_datapoints(points, y)

    for i in range(1000):
        indices = np.random.permutation(points.shape[0]) # random indices
        points = points[indices]
        y = y[indices]

        output = model.forward(points)
        model.backward(y)

    model.visualize_hidden_units()
    visualize_datapoints(points, output)
