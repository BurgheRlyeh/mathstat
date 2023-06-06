import pathlib
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF


def edf(data, cdf, x, title):
    fig, axes = plt.subplots(1, len(data), figsize=(12, 5))
    fig.suptitle(title)

    for i, inf in enumerate(data):
        y1 = ECDF(inf)(x)
        y2 = cdf(x)
        axes[i].plot(x, y1)
        axes[i].plot(x, y2)
        axes[i].set_title(f"n = {len(inf)}")

    plt.savefig(pathlib.Path(f"../images/edf/{title}.png"))


if __name__ == "__main__":
    edf(
        [
            np.random.standard_normal(20),
            np.random.standard_normal(60),
            np.random.standard_normal(100)
        ],
        sps.norm.cdf,
        np.linspace(-4, 4, 100),
        "normal",
    )

    edf(
        [
            np.random.standard_cauchy(20),
            np.random.standard_cauchy(60),
            np.random.standard_cauchy(100)
        ],
        sps.cauchy.cdf,
        np.linspace(-4, 4, 100),
        "cauchy",
    )

    scale = 1.0 / np.sqrt(2.0)
    edf(
        [
            np.random.laplace(loc=0, scale=scale, size=20),
            np.random.laplace(loc=0, scale=scale, size=60),
            np.random.laplace(loc=0, scale=scale, size=100)
        ],
        lambda x: sps.laplace.cdf(x, loc=0, scale=scale),
        np.linspace(-4, 4, 100),
        "laplace",
    )

    edf(
        [
            np.random.poisson(lam=10, size=20),
            np.random.poisson(lam=10, size=60),
            np.random.poisson(lam=10, size=100)
        ],
        lambda x: sps.poisson.cdf(x, 10),
        np.linspace(6, 14, 100),
        "poisson"
    )

    low = -np.sqrt(3.0)
    high = np.sqrt(3.0)
    edf(
        [
            np.random.uniform(low=low, high=high, size=20),
            np.random.uniform(low=low, high=high, size=60),
            np.random.uniform(low=low, high=high, size=100),
        ],
        lambda x: sps.uniform.cdf(x, low, 2 * high),
        np.linspace(-4, 4, 100),
        "uniform",
    )