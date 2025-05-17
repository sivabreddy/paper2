import numpy as np
import random, math


def algm(w):
    # Convert weights to a list of NumPy arrays if they aren't already
    if not isinstance(w, list):
        w = list(w)

    # Flatten and normalize the weights
    flat_weights = [arr.flatten() for arr in w]

    # Ensure consistent dimensionality
    Xmin, Xmax = 1, 5
    N, M = len(flat_weights), max(len(arr) for arr in flat_weights)

    alpha = 1
    r = random.random()
    l1, l2, l3 = 5 * math.exp(-2), 100, 1 * math.exp(-2)

    # Find global min and max across all weight arrays
    lb = min(arr.min() for arr in w)
    ub = max(arr.max() for arr in w)

    Tmax = 10

    def generate(N, M):
        return [[random.random() for _ in range(M)] for _ in range(N)]

    def bound(arr):
        return [[random.uniform(lb, ub) if val < 0 or val > 100 else val
                 for val in row] for row in arr]

    def fitness(soln):
        return [sum(row) + random.random() for row in soln]

    # Pad shorter arrays to match the longest array
    X = []
    for _ in range(N):
        row = [random.random() for _ in range(M)]
        X.append(row)

    Hj, Pij, Cj = l1 * random.random(), l2 * random.random(), l3 * random.random()
    F = random.uniform(-1, 1)
    Fit = fitness(X)
    Fbest = max(Fit)
    best = Fit.index(Fbest)
    Xbest = max(X[best])

    t = 0
    g = 1  # Absorption coefficient
    E = random.sample(range(1, len(w) + 1), len(X))

    while t < Tmax:
        beta, epsilon, T_teta, K = 0.5, 0.05, 298.15, 0.5
        T = math.exp(-t / Tmax)  # Temperature
        Hj = Hj * math.exp(-Cj * (1 / T) - (1 / T_teta))  # Henry's coefficient (eq.8)
        S = K * Hj * Pij  # solubility (eq.9)

        rr = np.sqrt(np.sum((X[0][0] - X[1][1]) ** 2))
        beta0 = math.exp(-g * rr)

        new_X = []
        for i in range(len(X)):
            tem = []
            for j in range(len(X[i])):
                Gamma = beta * math.exp(-(Fbest + epsilon) / (Fit[i] + epsilon))
                ############ proposed updated equation ##############
                n = ((beta0 * math.exp(-Gamma * r ** 2) * X[i][j] * alpha * E[i]) * (
                            (F * r * Gamma) + ((F * r * alpha) - 1)) + (F * r * (Gamma * Xbest + alpha * S * Xbest)) * (
                                 1 - beta0 * math.exp(-Gamma * r ** 2))) / (
                                (F * r * Gamma) + (F * r * alpha) - beta0 * math.exp(-Gamma * r ** 2))
                tem.append(n)
            new_X.append(tem)

        X = bound(new_X)

        c1, c2 = 0.1, 0.2
        Nw = M * (random.uniform(c1, c2) + c1)  # eq.11
        G = Xmin + r * (Xmax - Xmin)
        worst = round(G)

        Fit = fitness(X)
        Fbest = max(Fit)
        best = Fit.index(Fbest)
        Xbest = max(X[best])
        t += 1

    # Reconstruct the weight arrays with the best solution
    updated_weights = []
    start = 0
    for original_weight in w:
        shape = original_weight.shape
        size = np.prod(shape)

        # Ensure we have enough data to reshape
        if start + size > len(X[best]):
            # If not enough data, pad with the original weight values
            best_solution_slice = list(original_weight.flatten())
        else:
            best_solution_slice = X[best][start:start + size]

        updated_weight = np.array(best_solution_slice[:size]).reshape(shape)
        updated_weights.append(updated_weight)
        start += size

    return updated_weights