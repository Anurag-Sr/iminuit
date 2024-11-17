from iminuit import Minuit
from iminuit.cost import LeastSquares

import matplotlib.pyplot as plt

import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import chisquare

N = 10


def line(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def chisqu(obs: np.ndarray,
           exp: np.ndarray,
           error: np.ndarray) -> float:
    return np.sum((obs - exp) ** 2 / (error ** 2))


def main() -> None:

    # x = np.random.random(N)
    # y_err = np.random.random(N)
    # y = -line(x, 3.2, 10) + y_err

    x = np.array([0, 4, 8, 12, 16, 20, 24, 28])
    y = np.array([1360, 1300, 1240, 1160, 1100, 1020, 960, 900])
    y_err = np.ones(len(y)) * 12.5

    popt, pcov = curve_fit(line, x, y)
    chisq, _ = chisquare(y, line(x, popt[0], popt[1]))
    print("=========== Using scipy curve fit we get the following results ======== ")
    print('slope and constant ======== ', popt, '\n',
          'covariance matrix ========', pcov)
    print("chisq using scipy =====================", chisq)

    print("chisq not using scipy ==========",
          chisqu(y, line(x, popt[0], popt[1]), y_err))

    least_squares = LeastSquares(x, y, y_err, line)
    m = Minuit(least_squares, a=3, b=3)

    print("=====================Minuit results======================")
    print(m.migrad())
    print("chi square ===================== ", m.fval)
    print("chisq not using minuit ==========",
          chisqu(y, line(x, m.values[0], m.values[1]), y_err))
    print("reducuced chisq ================",
          chisqu(y, line(x, m.values[0], m.values[1]), y_err) / m.ndof)

    plt.errorbar(x, y, y_err, fmt=" ", marker='o')
    plt.plot(x, line(x, popt[0], popt[1]), label="scipy fit")
    plt.plot(x, line(x, m.values[0], m.values[1]), label="minuit fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()