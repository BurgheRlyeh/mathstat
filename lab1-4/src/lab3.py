import pathlib
import seaborn
import numpy as np
import matplotlib.pyplot as plt


def boxplot(data, title):
    nums = [str(len(inf)) for inf in data]
    fig, ax = plt.subplots(1, 1)
    seaborn.boxplot(data=data, orient="h", ax=ax)
    ax.set(xlabel="x", ylabel="n")
    ax.set(yticklabels=nums)
    ax.set_title(title)
    plt.savefig(pathlib.Path(f"../images/boxplots/{title}.png"))


def outlier(data):
    num = 0

    for i in range(1000):
        q1 = np.quantile(data, 0.25)
        q3 = np.quantile(data, 0.75)
        x1 = q1 - 1.5 * (q3 - q1)
        x2 = q1 + 1.5 * (q3 - q1)
        num += np.count_nonzero((data < x1) | (x2 < data)) / len(data)

    return round(num / 1000, 2)


if __name__ == "__main__":
    outliers = dict()

    d20, d100 = np.random.standard_normal(20), np.random.standard_normal(100)
    boxplot([d20, d100], "normal")
    outliers["normal"] = {"20": outlier(d20), "100": outlier(d100)}

    d20, d100 = np.random.standard_cauchy(20), np.random.standard_cauchy(100)
    boxplot([d20, d100], "cauchy")
    outliers["cauchy"] = {"20": outlier(d20), "100": outlier(d100)}

    scale = 1.0 / np.sqrt(2.0)
    d20, d100 = np.random.laplace(loc=0, scale=scale, size=20), np.random.laplace(loc=0, scale=scale, size=100)
    boxplot([d20, d100], "laplace")
    outliers["laplace"] = {"20": outlier(d20), "100": outlier(d100)}

    d20, d100 = np.random.poisson(lam=10, size=20), np.random.poisson(lam=10, size=100)
    boxplot([d20, d100], "poisson")
    outliers["poisson"] = {"20": outlier(d20), "100": outlier(d100)}

    low = -np.sqrt(3.0)
    high = np.sqrt(3.0)
    d20, d100 = np.random.uniform(low=low, high=high, size=20), np.random.uniform(low=low, high=high, size=100)
    boxplot([d20, d100], "uniform")
    outliers["uniform"] = {"20": outlier(d20), "100": outlier(d100)}

    # print(outliers)

