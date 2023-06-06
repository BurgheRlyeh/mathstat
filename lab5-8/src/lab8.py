import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path


def dispersion_exp(sample):
    return np.mean(sample ** 2) - np.mean(sample) ** 2


def normal(size):
    return np.random.standard_normal(size=size)


def draw_results(x_set, m_all, s_all, number):
    fig, axes = plt.subplots(1, 4)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0.9, 1.4)
    axes[3].set_ylim(0.9, 1.4)

    axes[0].hist(
        x_set[0],
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="N(0, 1) hyst n=20",
        color="black",
    )
    axes[0].legend(loc="best", frameon=True)
    axes[1].hist(
        x_set[1],
        density=True,
        histtype="stepfilled",
        alpha=0.3,
        label="N(0, 1) hyst n=100",
        color="black",
    )
    axes[1].legend(loc="best", frameon=True)

    axes[2].plot(m_all[0], [1, 1], label='"m" interval n = 20', color="lightcoral")
    axes[2].plot(m_all[1], [1.1, 1.1], label='"m" interval n = 100', color="steelblue")
    axes[2].legend()

    axes[3].plot(s_all[0], [1, 1], label="sigma interval n = 20", color="lightcoral")
    axes[3].plot(s_all[1], [1.1, 1.1], label="sigma interval n = 100", color="steelblue")
    axes[3].legend()

    fig.savefig(Path(r"../images/interval/8_{}".format(number)), dpi=300)

    plt.show()


if __name__ == "__main__":
    n_set = [20, 100]
    x_20 = normal(20)
    x_100 = normal(100)
    x_set = [x_20, x_100]

    alpha = 0.05
    m_all = []
    s_all = []
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = np.mean(x)
        s = np.sqrt(dispersion_exp(x))

        t_value = stats.t.ppf(1 - alpha / 2, n - 1)
        chi2_value = stats.chi2.ppf(1 - alpha / 2, n - 1)

        m1 = [
            m - s * t_value / np.sqrt(n - 1),
            m + s * t_value / np.sqrt(n - 1)
        ]
        s1 = [
            s * np.sqrt(n) / np.sqrt(chi2_value),
            s * np.sqrt(n) / np.sqrt(stats.chi2.ppf(alpha / 2, n - 1))
        ]

        m_all.append(m1)
        s_all.append(s1)

        print("n: {}".format(n))
        print("m: {:.2f}, {:.2f}".format(m1[0], m1[1]))
        print("sigma: {:.2f}, {:.2f}".format(s1[0], s1[1]))

    draw_results(x_set, m_all, s_all, 1)
    print()

    m_all = []
    s_all = []
    for i in range(len(n_set)):
        n = n_set[i]
        x = x_set[i]

        m = np.mean(x)
        s = np.sqrt(dispersion_exp(x))

        norm_value = stats.norm.ppf(1 - alpha / 2)
        e = (np.sum((x - m) ** 4) / n) / s ** 4 - 3

        m_as = [
            m - norm_value / np.sqrt(n),
            m + norm_value / np.sqrt(n)
        ]
        s_as = [
            s / np.sqrt(1 + norm_value * np.sqrt((e + 2) / n)),
            s / np.sqrt(1 - norm_value * np.sqrt((e + 2) / n))
        ]

        m_all.append(m_as)
        s_all.append(s_as)

        print("m asymptotic: {:.2f}, {:.2f}".format(m_as[0], m_as[1]))
        print("sigma asymptotic: {:.2f}, {:.2f}".format(s_as[0], s_as[1]))

    draw_results(x_set, m_all, s_all, 2)
