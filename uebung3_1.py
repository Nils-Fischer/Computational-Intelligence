import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import ndarray


def fermi_function(a: ndarray):
    return 1.0 / (1.0 + np.exp(-a))


def fermi_function_derivation(a: ndarray):
    return np.exp(-a) / ((1 + np.exp(-a)) ** 2)


def reLu(x):
    if x < 0:
        return -1
    else:
        return 1


def reLu_derivation(x):
    if x == 0:
        return np.nan  # undefined
    elif x < 0:
        return -1
    else:
        return 1


def linear(a: ndarray):
    return a


def linear_derivation(a: ndarray):
    return np.ones_like(a)


class MLP():
    def __init__(self, input_size, output_size, activation_fn_hid, activation_fn_hid_derivation, activation_fn_out, activation_fn_out_derivation):
        self.activation_fn_hid = activation_fn_hid
        self.activation_fn_hid_derivation = activation_fn_hid_derivation
        self.activation_fn_out = activation_fn_out
        self.activation_fn_out_derivation = activation_fn_out_derivation

        # +1 bias weight
        self.weights_hid = np.random.uniform(-1, 1, (20, input_size + 1))
        self.weights_out = np.random.uniform(-1, 1, 20 + 1)  # +1 bias weight

    def forward(self, x: np.ndarray):
        # weights_hid Matrix (20x2) matmul o_in Vektor (2x1) = a_hid Vektor (20x1)
        self.o_in = np.vstack(([1], x[:, None]))
        self.a_hid = self.weights_hid @ self.o_in  # input to hidden layer
        o_hid = self.activation_fn_hid(self.a_hid)  # activation function

        bias = np.asarray([[1]])
        # o_hid Vektor (21x1), o_hid[0] = 1 :bias
        self.o_hid = np.concatenate((bias, o_hid))

        # weights_out Matrix (1x21) matmul o_hid Vektor (21x1) = a_out Vektor (1x1)
        self.a_out = self.weights_out @ self.o_hid
        o_out = self.activation_fn_out(self.a_out)  # Linear
        return o_out.item()

    def train(self, x, t, lernrate):
        y = self.forward(x)

        delta_out = self.activation_fn_out_derivation(
            self.a_out) * (y - t)  # = (y - t) cause derivation of Linear = 1
        # activation_fn_hid_derivation(self.a_hid).T Matrix (1x20) elementwise_mul weights_out[1:] (1x20) * delta_out (scalar) = delta_hid Matrix (1x20)
        delta_hid = self.activation_fn_hid_derivation(
            self.a_hid).T * self.weights_out[1:] * delta_out

        # o_hid Matrix (21x1) * delta_out (scalar) = delta_w_out Matrix (21x1)
        delta_w_out = -lernrate * self.o_hid * delta_out
        # delta_hid.T Matrix (20x1) matmul o_in.T Matrix (1x2) = delta_w_hid Matrix (20x2)
        delta_w_hid = -lernrate * delta_hid.T @ self.o_in.T

        self.weights_hid += delta_w_hid
        self.weights_out += delta_w_out.flatten()

    def training(self, input, expected, epochs=1000, learnrate=0.1):
        error = []
        for epoch in range(epochs):
            # shuffling
            indices = np.random.permutation(input.shape[0])
            input = input[indices]
            expected = expected[indices]

            for x, t in zip(input, expected):
                self.train(x, t, learnrate)
            sum = 0
            for x, t in zip(input, expected):
                sum += (self.forward(x) - t)**2
            error.append(sum/input.shape[0])
            print(f"Epoche: {epoch+1} MSE: {sum/input.shape[0]}")
        return error


def visualize_datapoints(points):
    A = points[points[:, -1] >= 0][:, :2].T
    B = points[points[:, -1] < 0][:, :2].T

    fix, ax = plt.subplots()
    ax.scatter(A[0], A[1], color="red", label="A")
    ax.scatter(B[0], B[1], color="blue", label="B")
    ax.set_xlim(np.min(np.append(A[0], B[0])) *
                1.1, np.max(np.append(A[0], B[0]))*1.1)
    ax.set_ylim(np.min(np.append(A[1], B[1])) *
                1.1, np.max(np.append(A[1], B[1]))*1.1)
    ax.set_xlabel("x_1")
    ax.set_ylabel("x_2")
    ax.legend()
    plt.show()


if __name__ == '__main__':
    np.random.seed(2423)
    u = np.arange(1, 251)[:, None]  # shape of a vector
    x_A1 = 1 + np.sin(0.1*u+1.5)*np.sqrt(u+10)
    x_A2 = -2 + np.cos(0.1*u+1.5)*np.sqrt(u+10)
    y_A = np.ones_like(u)
    a = np.hstack((x_A1, x_A2, y_A))

    x_B1 = 1 + np.sin(0.1*u-1.6)*np.sqrt(u+10)
    x_B2 = -2 + np.cos(0.1*u-1.6)*np.sqrt(u+10)
    y_B = - np.ones_like(u)
    b = np.hstack((x_B1, x_B2, y_B))

    points = np.vstack((a, b))
    # visualize_datapoints(points)

    model = MLP(2, 1, fermi_function, fermi_function_derivation,
                linear, linear_derivation)
    error = model.training(points[:, :2], points[:, -1], epochs=100)
    result = np.array([model.forward(p) for p in points[:, :2]])[:, None]

    visualize_datapoints(np.hstack((points[:, :2], result)))
