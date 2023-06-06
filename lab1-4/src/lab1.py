import math
import pathlib
import seaborn
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def normal(sizes=(10, 50, 1000)):
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(r"Случайная величина $\xi \sim \mathcal{N}(0, 1)$")

    for i in range(len(sizes)):
        distribution = np.random.standard_normal(size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(distribution, bins=30, density=True, alpha=0.6, label="Гистограмма выборки")
        plt.plot(grid, sps.norm.pdf(grid), color="red", lw=3, label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(pathlib.Path("../images/histogram/normal.png"))


def cauchy(sizes=(10, 50, 1000)):
    grid = np.linspace(-30, 30, 1000)
    plt.figure(figsize=(15, 5)).suptitle(r"Случайная величина $\xi \sim \mathcal{C}(0, 1)$")

    for i in range(len(sizes)):
        distribution = sps.cauchy.rvs(loc=0, scale=1, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.xlim([-10, 10])
        seaborn.histplot(distribution, kde=False, stat="density", label="samples")
        plt.plot(grid, sps.cauchy.pdf(grid), color="red", lw=3, label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(pathlib.Path("../images/histogram/cauchy.png"))


def laplace(sizes=(10, 50, 1000)):
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(r"Случайная величина $\xi \sim \mathcal{L}(0, 1/\sqrt{2})$")

    for i in range(len(sizes)):
        distribution = np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(distribution, bins=30, density=True, alpha=0.6, label="Гистограмма выборки")
        plt.plot(grid, sps.laplace.pdf(grid, loc=0, scale=1.0 / np.sqrt(2.0)), color="red", lw=3,
                 label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(pathlib.Path("../images/histogram/laplace.png"))


def poisson(sizes=(10, 50, 1000)):
    grid = np.linspace(0, 20, 1000)
    plt.figure(figsize=(15, 5)).suptitle(r"Случайная величина $\xi \sim \mathcal{P}(10)$")

    for i in range(len(sizes)):
        distribution = np.random.poisson(lam=10, size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(distribution, bins=30, density=True, alpha=0.6, label="Гистограмма выборки")
        y = [(10 ** x * np.exp(-10) / math.gamma(x + 1)) for x in grid]
        plt.plot(grid, y, color="red", lw=3, label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(pathlib.Path("../images/histogram/poisson.png"))


def uniform(sizes=(10, 50, 1000)):
    grid = np.linspace(-3, 3, 1000)
    plt.figure(figsize=(15, 5)).suptitle(r"Случайная величина $\xi \sim \mathcal{U}(-\sqrt{3}, \sqrt{3})$")

    for i in range(len(sizes)):
        distribution = np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=sizes[i])
        plt.subplot(1, 3, i + 1)
        plt.hist(distribution, bins=30, density=True, alpha=0.6, label="Гистограмма выборки")
        plt.plot(grid, sps.uniform.pdf(grid, loc=-np.sqrt(3.0), scale=2 * np.sqrt(3.0)), color="red", lw=3,
                 label="Плотность случайной величины")
        plt.title(f"\nРазмер выборки: {sizes[i]}", fontsize=10)

    plt.legend(fontsize=10, loc=1)
    plt.savefig(pathlib.Path("../images/histogram/uniform.png"))


if __name__ == "__main__":
    normal()
    cauchy()
    laplace()
    poisson()
    uniform()

