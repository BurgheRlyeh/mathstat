import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pathlib import Path


def func(x):
    return 2 + 2 * x


def noise_func(x):
    return func(x) + np.random.normal(0, 1, len(x))


def LMM(parameters, x, y):
    alpha_0, alpha_1 = parameters
    return np.sum(np.abs(y - alpha_0 - alpha_1 * x))


def get_MNK_params(x, y):
    beta_1 = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (
        np.mean(x * x) - np.mean(x) ** 2
    )
    beta_0 = np.mean(y) - beta_1 * np.mean(x)
    return beta_0, beta_1


def get_MNM_params(x, y):
    beta_0, beta_1 = get_MNK_params(x, y)
    result = opt.minimize(LMM, [beta_0, beta_1], args=(x, y), method="SLSQP")
    alpha_0, alpha_1 = result.x
    return alpha_0, alpha_1


def MNK(x, y):
    beta_0, beta_1 = get_MNK_params(x, y)
    print("beta_0 =", beta_0, "beta_1 =", beta_1)
    y_new = beta_0 + beta_1 * x
    return y_new


def MNM(x, y):
    alpha_0, alpha_1 = get_MNM_params(x, y)
    print("alpha_0 =", alpha_0, "alpha_1 =", alpha_1)
    y_new = alpha_0 + alpha_1 * x
    return y_new


def get_dist(y_model, y_regr):
    dist_y = np.sum((y_model - y_regr) ** 2)
    return dist_y


def plot_lin_regression(text, x, y, number):
    y_mnk = MNK(x, y)
    y_mnm = MNM(x, y)
    y_dist_mnk = get_dist(y, y_mnk)
    y_dist_mnm = get_dist(y, y_mnm)
    print("MNK distance:", y_dist_mnk)
    print("MNM distance:", y_dist_mnm)
    plt.scatter(x, y, label="Выборка", color="black", marker=".", linewidths=0.7)
    plt.plot(x, func(x), label="Модель", color="lightcoral")
    plt.plot(x, y_mnk, label="МНК", color="steelblue")
    plt.plot(x, y_mnm, label="МНМ", color="lightgreen")
    plt.xlim([-1.8, 2])
    plt.grid()
    plt.legend()
    plt.savefig(Path(f"../images/regression/6_{number}"))
    plt.show()


if __name__ == "__main__":
    x = np.arange(-1.8, 2, 0.2)
    y = noise_func(x)
    plot_lin_regression("NoPerturbations", x, y, 1)

    x = np.arange(-1.8, 2, 0.2)
    y = noise_func(x)
    y[0] += 10
    y[-1] -= 10
    plot_lin_regression("Perturbations", x, y, 2)
