import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
from pathlib import Path

eps = 10e-4

def plot_data(data):
    """
    Plot the experiment data.
    """
    plt.figure()
    data.plot(color="green", linewidth=0.5)
    plt.title("Experiment data")
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(Path("../images/data.png"))

def diagram(data, epsilon, beta):
    """
    Create a diagram plot.
    """
    plt.figure()
    plt.fill_between(data.index, data - epsilon, data + epsilon, color="skyblue", alpha=0.3)
    plt.plot(data.index, data, color="green", linewidth=0.5)
    if beta is not None:
        plt.plot([0, 199], [beta, beta], color="maroon", linestyle="--", linewidth=0.5)
    plt.title("Diagram")
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(Path(f"../images/diagram_beta_{beta}.png"))

def diagram_with_mode(data, mode):
    """
    Create a diagram plot with mode.
    """
    data_mode, data_not_mode = [], []
    index_mode, index_not_mode = [], []
    for i, d in enumerate(data):
        in_mode = False
        for m in mode:
            if m[0] >= d - eps and m[1] <= d + eps:
                data_mode.append(d)
                index_mode.append(i)
                in_mode = True
                break
        if not in_mode:
            data_not_mode.append(d)
            index_not_mode.append(i)
    plt.figure()
    plt.fill_between(index_mode, np.array(data_mode) - eps, np.array(data_mode) + eps, color="lightcoral", alpha=0.3)
    plt.plot(index_mode, data_mode, color="brown", linewidth=0.5)
    plt.fill_between(index_not_mode, np.array(data_not_mode) - eps, np.array(data_not_mode) + eps, color="skyblue", alpha=0.3)
    plt.plot(index_not_mode, data_not_mode, color="blue", linewidth=0.5)
    for m in mode:
        plt.plot([0, 199], [m[0], m[0]], color="maroon", linestyle="--", linewidth=0.5)
        plt.plot([0, 199], [m[1], m[1]], color="maroon", linestyle="--", linewidth=0.5)
    plt.title("Diagram with mode")
    plt.xlabel("n")
    plt.ylabel("mV")
    plt.savefig(Path("../images/diagram_with_mode.png"))

def estimators_of_data(data):
    """
    Calculate the estimators of data.
    """
    size = len(data)
    return data[0] - eps, data[size - 1] + eps

def mode_and_max_click(data):
    """
    Find the mode and maximum number of clicks.
    """
    y = []
    for element in data:
        y.append(element - eps)
        y.append(element + eps)
    y = list(set(y))
    y.sort()
    z = []
    for i in range(len(y) - 1):
        z.append([y[i], y[i + 1]])
    max_mu = 0
    coefs = []
    for i in range(len(z)):
        mu = sum(1 for d in data if z[i][0] <= d - eps <= z[i][1] or z[i][0] <= d + eps <= z[i][1])
        if mu > max_mu:
            max_mu = mu
            coefs = []
            coefs.append(i)
        if mu == max_mu:
            coefs.append(i)
    mode = [z[i] for i in range(len(z)) if i in coefs]
    return mode, max_mu

def Jakar_coeff(int_data):
    """
    Calculate the Jaccard coefficient.
    """
    min_inc = [int_data[0][0], int_data[0][1]]
    for interval in int_data:
        min_inc[0] = max(min_inc[0], interval[0])
        min_inc[1] = min(min_inc[1], interval[1])
    max_inc = [int_data[0][0], int_data[-1][1]]
    JK = (min_inc[1] - min_inc[0]) / (max_inc[1] - max_inc[0])
    return JK

def relative_width_of_the_mode(int_data, mode):
    """
    Calculate the relative width of the mode.
    """
    wid_mode = sum(m[1] - m[0] for m in mode)
    return wid_mode / (int_data[-1][1] - int_data[0][0])

def find_oskorbin_center_and_w(data):
    """
    Find the Oskorbin center and width.
    """
    A, b = [], []
    n = len(data)
    for i in range(n):
        A.append([-eps, -1])
        A.append([-eps, 1])
    A.append([-1, 0])
    for d in data:
        b.append(-d)
        b.append(d)
    b.append(-1)
    c = [1, 0]
    result = opt.linprog(c=c, A_ub=A, b_ub=b).x
    w, beta = result
    return w, beta

if __name__ == "__main__":
    data = pd.read_csv("../src/data.csv", sep=";", encoding="cp1251")
    print(data)
    data = data["mB"]
    interval_data = [[d - eps, d + eps] for d in data]
    plot_data(data)
    print(f"estimators of data = {estimators_of_data(data)}")
    mode, max_mu = mode_and_max_click(data)
    print(f"mode = {mode}")
    print(f"max_mu = {max_mu}")
    diagram(data, eps, None)
    diagram_with_mode(data, mode)
    w, beta = find_oskorbin_center_and_w(data)
    print(f"[w, beta] = {[w, beta]}")
    diagram(data, eps * w, beta)
    Ji = Jakar_coeff(interval_data)
    print(f"Ji = {Ji}")
    ro = relative_width_of_the_mode(interval_data, mode)
    print(f"ro = {ro}")
