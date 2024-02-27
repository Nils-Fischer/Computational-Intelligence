import numpy as np
from itertools import product


def step_func(z):
    return 1.0 if (z >= 0) else 0.0


def correct(w, z, t, y):
    return w + z * (t - y)


def perceptron(input, w, expected):
    mistakes = len(w)
    epoch = 0

    while mistakes != 0:
        mistakes = 0
        print(f"|{epoch}", end="|")

        print(f"{w}", end="|")
        for i, z in enumerate(input):
            result = step_func(np.dot(z, w))
            t = expected[i]
            mistakes += int((result - t) ** 2)
            w = correct(w, z, t, result)
        print(f"{mistakes}", end="|\n")
        epoch += 1
    print(f"Nach dem Lernen: {w}")


def main():
    inputs = [np.array(a) for a in list(product([0, 1], [0, 1], [0, 1]))]
    extended_inputs = np.array(
        [
            np.array([1, x1, x2, x3, x1 & x2, x2 & x3, x1 & x3, x1 & x2 & x3])
            for (x1, x2, x3) in inputs
        ]
    )
    weights = np.zeros((8,))
    expected_output = np.array([1, 0, 0, 1, 0, 1, 1, 0])
    perceptron(extended_inputs, weights, expected_output)


if __name__ == "__main__":
    main()
