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
          err: np.ndarray) -> float:
    return np.sum(((obs - exp) ** 2)/(err ** 2))


def main() -> None:

    x = np.random.random(N)
    y_err = np.random.random(N)
    y = -line(x, 3.2, 10) + y_err

    popt, pcov = curve_fit(line, x, y)
    chisq = chisquare(y, line(x, popt[0], popt[1]))
    print("=========== Using scipy curve fit we get the following results ======== ")
    print(popt, pcov)
    print("chisq using scipy =====================", chisq[0])

    print("chisq not using scipy ==========",
          chisqu(y, line(x, popt[0], popt[1]), y_err))

    least_squares = LeastSquares(x, y, y_err, line)
    m = Minuit(least_squares, a=3, b=3)

    print("=====================Minuit results======================")
    print(m.migrad())
    print(m.hesse())
    print("chi square ===================== ", m.fval)
    print("chisq not using minuit ==========",
          chisqu(y, line(x, m.values[0], m.values[1]), y_err))

    plt.errorbar(x, y, y_err, fmt=" ", marker='o')
    plt.plot(x, line(x, popt[0], popt[1]), label = "scipy fit")
    plt.plot(x, line(x, m.values[0], m.values[1]), label = "minuit fit")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()