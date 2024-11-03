import minimization
import classes
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit # type: ignore
import argparse
import logging

points  = 50000
bins = 1000

# Create and configure logger
logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.INFO)

def gaussian(mean,std):
    pdf = classes.Gaussian(mean, std)
    for i in range (points):
        _ = pdf.next()

    data = pdf.Mass()

    x = np.linspace(pdf.XMIN, pdf.XMAX, num=10000, endpoint=True,)

    minimiser = minimization.NegativeLogLikelihood(pdf,data)
    minimisation = Minuit(minimiser.evaluate_gaussian, mean = 1.4, sigma = 0.1)
    minimisation.errordef = 0.5

    # obtain the minimisation results and print
    result = minimisation.migrad()
    # Output minimisation results
    print(result)
    logger.info("fit results for gaussian %s", str(result.values))

    vals = np.linspace(0,10,100,endpoint = True)
    fit = pdf.evaluate(vals) / pdf.integrate((pdf.XMIN, pdf.XMAX))
    plt.hist(data, bins, label="generated dataset", alpha = 0.3, density=True) 
    plt.plot(vals, fit, linestyle="--", color="red",alpha = 0.7 ,label="optimum Fit")
    plt.plot(x, pdf.evaluate(x)/pdf.integrate((pdf.XMIN, pdf.XMAX)), label = 'fit')
    plt.xlim(0,3)
    plt.xlabel("Mass")
    plt.ylabel("events")
    plt.legend()
    plt.show()

def exponential(decay_constant):
    
    pdf = classes.Exponential(decay_constant)
    for i in range (points):
        _ = pdf.next()

    data = pdf.Mass()
    x = np.linspace(pdf.XMIN, pdf.XMAX, num=10000, endpoint=True,)
    minimiser = minimization.NegativeLogLikelihood(pdf,data)
    minimisation = Minuit(minimiser.evaluate_exponential, decay_constant = 2)
    minimisation.errordef = 0.5

    # obtain the minimisation results and print
    result = minimisation.migrad()
    # Output minimisation results
    print(result)
    logger.info("fit results for exponential %s", str(result.values))
    
    vals = np.linspace(0,10,100,endpoint = True)
    fit = pdf.evaluate(vals) / pdf.integrate((pdf.XMIN, pdf.XMAX))
    plt.hist(data, bins=1000, label="Data", alpha = 0.3, density=True) 
    plt.plot(x, pdf.evaluate(x)/pdf.integrate((pdf.XMIN, pdf.XMAX)), label = 'fit')
    plt.plot(vals, fit, linestyle="--", color="red",alpha = 0.7 ,label="optimum Fit")
    plt.xlim(pdf.XMIN, pdf.XMAX)
    plt.xlabel("Mass")
    plt.ylabel("events")
    plt.legend()
    plt.show()


plt.title('Generated Exponential dataset')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dist", help="distribution type")
    parser.add_argument("-m", "--mean", help="mean")
    parser.add_argument("-s", "--std", help="std")
    parser.add_argument("-dc", "--decay", help="eponential decay constant")
    options = parser.parse_args()
    print(options)

    if options.dist == "expo":
        exponential(float(options.decay))
    
    elif options.dist == "Gauss":
        gaussian(float(options.mean),float(options.std))
    
    else:
        print ("please provide a distribution for the data")

if __name__ == "__main__":
    main()