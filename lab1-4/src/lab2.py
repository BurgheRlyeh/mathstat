import numpy as np


def get_characteristics(generator, sample_size):
    iters = 1000
    cs = dict()
    for num in sample_size:
        cs[num] = dict()
        mean = []
        median = []
        z_r = []
        z_q = []
        z_tr = []
        for _ in range(iters):
            data = generator(num)
            data.sort()

            mean.append(data.mean())
            median.append(np.median(data))
            z_r.append((data[0] + data[-1]) / 2)
            z_q.append((np.quantile(data, 0.25) + np.quantile(data, 0.75)) / 2)
            r = num // 4
            z_tr.append(sum(data[r:-r]) / (num - 2 * r))

        cs[num]["mean"] = round(np.mean(mean), 4)
        cs[num]["median"] = round(np.mean(median), 4)
        cs[num]["z_R"] = round(np.mean(z_r), 4)
        cs[num]["z_Q"] = round(np.mean(z_q), 4)
        cs[num]["z_tr"] = round(np.mean(z_tr), 4)
        cs[num]["d_mean"] = round(np.std(mean) ** 2, 4)
        cs[num]["d_median"] = round(np.std(median) ** 2, 4)
        cs[num]["d_z_R"] = round(np.std(z_r) ** 2, 4)
        cs[num]["d_z_Q"] = round(np.std(z_q) ** 2, 4)
        cs[num]["d_z_tr"] = round(np.std(z_tr) ** 2, 4)

        cs[num]["mean+"] = (
                "["
                + str(round(np.mean(mean) - np.std(mean), 4))
                + "; "
                + str(round(np.mean(mean) + np.std(mean), 4))
                + "]"
        )
        cs[num]["median+"] = (
                "["
                + str(round(np.mean(median) - np.std(median), 4))
                + "; "
                + str(round(np.mean(median) + np.std(median), 4))
                + "]"
        )
        cs[num]["z_R+"] = (
                "["
                + str(round(np.mean(z_r) - np.std(z_r), 4))
                + "; "
                + str(round(np.mean(z_r) + np.std(z_r), 4))
                + "]"
        )
        cs[num]["z_Q+"] = (
                "["
                + str(round(np.mean(z_q) - np.std(z_q), 4))
                + "; "
                + str(round(np.mean(z_q) + np.std(z_q), 4))
                + "]"
        )
        cs[num]["z_tr+"] = (
                "["
                + str(round(np.mean(z_tr) - np.std(z_tr), 4))
                + "; "
                + str(round(np.mean(z_tr) + np.std(z_tr), 4))
                + "]"
        )
    return cs


def chars():
    sample_size = (10, 100, 1000)
    cs = dict()
    cs["normal"] = get_characteristics(np.random.standard_normal, sample_size)

    cs["cauchy"] = get_characteristics(np.random.standard_cauchy, sample_size)
    cs["laplace"] = get_characteristics(
        lambda n: np.random.laplace(loc=0, scale=1.0 / np.sqrt(2.0), size=n),
        sample_size,
    )
    cs["poisson"] = get_characteristics(
        lambda n: np.random.poisson(lam=10, size=n), sample_size
    )
    cs["uniform"] = get_characteristics(
        lambda n: np.random.uniform(low=-np.sqrt(3.0), high=np.sqrt(3.0), size=n),
        sample_size,
    )
    return cs


if __name__ == "__main__":
    characteristics = chars()
    for dist in ["normal", "cauchy", "laplace", "poisson", "uniform"]:
        print(dist)
        print(r"\begin{table}")
        print(rf"\caption{{dist}}")
        for i in (10, 100, 1000):
            v = characteristics[dist][i]
            print(r"\begin{adjustbox}{width=\textwidth}")
            print(r"\begin{tabular}{| c | c | c | c | c | c |}")
            print(r"\hline")
            print(f"n = {i}" + r" & $\bar{x}$ & $med x$ & $z_R$ & $z_Q$ & $z_{tr}$ \\\hline")
            print("$E(x)$ &", end=" ")
            print(f"{v['mean']} & {v['median']} & {v['z_R']} & {v['z_Q']} & {v['z_tr']}", end=" ")
            print(r"\\\hline")
            print("$D(x)$ &", end=" ")
            print(f"{v['d_mean']} & {v['d_median']} & {v['d_z_R']} & {v['d_z_Q']} &  {v['d_z_tr']}", end=" ")
            print(r"\\\hline")
            print(r"$E(x) \pm \sqrt{D(x)}$ &", end=" ")
            print(f"{v['mean+']} & {v['median+']} & {v['z_R+']} & {v['z_Q+']} & {v['z_tr+']}" r"\\\hline")
            print(r"\end{tabular}")
            print(r"\end{adjustbox}")
            print()
        print(r"\end{table}")

