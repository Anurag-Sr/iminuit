import minimization
import classes
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit  # type: ignore
import definitions

POINTS = 50000
BINS = 1000
LOGGER = definitions.logger()


def gaussian(
        mean: float,
        std: float) -> None:

    pdf = classes.Gaussian(mean, std)
    for i in range(POINTS):
        _ = pdf.next()

    data = pdf.Mass()

    x = np.linspace(pdf.XMIN, pdf.XMAX, num=10000, endpoint=True,)

    minimiser = minimization.NegativeLogLikelihood(pdf, data)
    minimisation = Minuit(minimiser.evaluate_gaussian, mean=1.4, sigma=0.1)
    minimisation.errordef = 0.5

    # obtain the minimisation results and print
    result = minimisation.migrad()
    # Output minimisation results
    print(result)
    LOGGER.info("Gaussian fit results mean: %s with std: +/- %s", str(result.values[0]), str(result.errors[0]))

    vals = np.linspace(0, 10, 100, endpoint=True)
    fit = pdf.evaluate(vals) / pdf.integrate((pdf.XMIN, pdf.XMAX))
    plt.hist(data, BINS, label="generated dataset", alpha=0.3, density=True)
    plt.plot(vals, fit, linestyle="--", color="red", alpha=0.7, label="optimum Fit")
    plt.plot(x, pdf.evaluate(x) / pdf.integrate((pdf.XMIN, pdf.XMAX)), label='fit')
    plt.xlim(0, 3)
    plt.xlabel("Mass")
    plt.ylabel("events")
    plt.title("Gaussian dataset")
    plt.legend()
    plt.show()


def exponential(decay_constant: float) -> None:

    pdf = classes.Exponential(decay_constant)
    for i in range(POINTS):
        _ = pdf.next()

    data = pdf.Mass()
    x = np.linspace(pdf.XMIN, pdf.XMAX, num=10000, endpoint=True,)
    minimiser = minimization.NegativeLogLikelihood(pdf, data)
    minimisation = Minuit(minimiser.evaluate_exponential, decay_constant=2)
    minimisation.errordef = 0.5

    # obtain the minimisation results and print
    result = minimisation.migrad()
    # Output minimisation results
    print(result)
    LOGGER.info("exponential fit results decay constant: %s with error: +/- %s", str(result.values[0]), str(result.errors[0]))

    vals = np.linspace(0, 10, 100, endpoint=True)
    fit = pdf.evaluate(vals) / pdf.integrate((pdf.XMIN, pdf.XMAX))
    plt.hist(data, BINS, label="Data", alpha=0.3, density=True)
    plt.plot(x, pdf.evaluate(x) / pdf.integrate((pdf.XMIN, pdf.XMAX)), label='fit')
    plt.plot(vals, fit, linestyle="--", color="red", alpha=0.7, label="optimum Fit")
    plt.xlim(pdf.XMIN, pdf.XMAX)
    plt.xlabel("Mass")
    plt.ylabel("events")
    plt.legend()
    plt.title('Generated Exponential dataset')
    plt.show()


def linear(slope: float) -> None:

    pdf = classes.Linear(slope)
    for i in range(POINTS):
        _ = pdf.next()

    data = pdf.Mass()
    x = np.linspace(pdf.XMIN, pdf.XMAX, num=10000, endpoint=True,)
    minimiser = minimization.NegativeLogLikelihood(pdf, data)
    minimisation = Minuit(minimiser.evaluate_linear, slope=1.5)
    minimisation.errordef = 0.5

    # obtain the minimisation results and print
    result = minimisation.migrad()
    # Output minimisation results
    print(result)
    LOGGER.info("linear fit results slope: %s with error: +/- %s", str(result.values[0]), str(result.errors[0]))

    vals = np.linspace(0, 10, 100, endpoint=True)
    fit = pdf.evaluate(vals) / pdf.integrate((pdf.XMIN, pdf.XMAX))
    plt.plot(x, pdf.evaluate(x) / pdf.integrate((pdf.XMIN, pdf.XMAX)), label='fit')
    plt.hist(data, BINS, label="Data", alpha=0.3, density=True)
    plt.plot(vals, fit, linestyle="--", color="red", alpha=0.7, label="optimum Fit")
    plt.xlim(pdf.XMIN, pdf.XMAX)
    plt.xlabel("Mass")
    plt.ylabel("events")
    plt.legend()
    plt.title("Linear dataset")
    plt.show()


def main():
    parser = definitions.parser()
    options = parser.parse_args()
    print(options)

    if options.dist == "expo":
        exponential(float(options.decay))
    elif options.dist == "Gauss":
        gaussian(float(options.mean), float(options.std))
    elif options.dist == "Lin":
        linear(float(options.slope))
    else:
        print("please provide a distribution for the data")


if __name__ == "__main__":
    main()
