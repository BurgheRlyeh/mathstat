import pathlib
import seaborn
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt


def kde(data, pdf, x, title):
    scales = [0.5, 1.0, 2.0]
    fig, ax = plt.subplots(1, len(scales), figsize=(12, 4))
    fig.suptitle(f"{title}, n = {len(data)}")

    for i, s in enumerate(scales):
        seaborn.kdeplot(data, ax=ax[i], bw_method="silverman", bw_adjust=s, label="kde")
        ax[i].set_xlim([x[0], x[-1]])
        ax[i].set_ylim([0, 1])
        ax[i].plot(x, [pdf(xk) for xk in x], label="pdf")
        ax[i].set_ylabel("Плотность")
        ax[i].legend()
        ax[i].set_title(f"h={str(s)}*$h_n$")

    plt.savefig(pathlib.Path(f"../images/kde/{title}_{len(data)}.png"))


if __name__ == "__main__":
    for n in [20, 60, 100]:
        kde(
            np.random.standard_normal(n),
            sps.norm.pdf,
            np.linspace(-4, 4, 100),
            "normal",
        )

    for n in [20, 60, 100]:
        kde(
            np.random.standard_cauchy(n),
            sps.cauchy.pdf,
            np.linspace(-4, 4, 100),
            "cauchy",
        )

    scale = 1.0 / np.sqrt(2.0)
    for n in [20, 60, 100]:
        kde(
            np.random.laplace(loc=0, scale=scale, size=n),
            lambda x: sps.laplace.pdf(x, loc=0, scale=scale),
            np.linspace(-4, 4, 100),
            "laplace",
        )

    for n in [20, 60, 100]:
        kde(
            np.random.poisson(lam=10, size=n),
            lambda x: sps.poisson.pmf(x, 10),
            np.linspace(6, 14, 10),
            "poisson",
        )

    low = -np.sqrt(3.0)
    high = np.sqrt(3.0)
    for n in [20, 60, 100]:
        kde(
            np.random.uniform(low=low, high=high, size=n),
            lambda x: sps.uniform.pdf(x, low, 2 * high),
            np.linspace(-4, 4, 100),
            "uniform",
        )

