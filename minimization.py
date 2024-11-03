import numpy as np

class NegativeLogLikelihood(object):
    """
    Class containing minimisation statistic to be for pdf fitting. The class takes in data as the input.
    """

    def __init__(self, pdf, data):

        self.pdf = pdf
        self.data = data


    def evaluate_gaussian(self, mean, sigma):
        """
        Evaluate negative log likelihood for gaussian signal and linear background
        """

        #This is where the params def takes effect as we can now set different parameters as we want for the minimisation
        self.pdf.params(mean=mean, sigma=sigma)
        norm = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood = self.pdf.evaluate(self.data,)/norm 
        # fudge factor for negative or 0 values. These are set very close to 0
        if (evaluate_likelihood <= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5
        NLL = np.log(evaluate_likelihood)
        return -NLL.sum()

    def evaluate_exponential(self, decay_constant):
        """
        Evaluate negative log likelihood for gaussian signal and linear background
        """

        #This is where the params def takes effect as we can now set different parameters as we want for the minimisation
        self.pdf.params(decay_constant = decay_constant)
        norm = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood = self.pdf.evaluate(self.data,)/norm 
        # fudge factor for negative or 0 values. These are set very close to 0
        if (evaluate_likelihood <= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5
        NLL = np.log(evaluate_likelihood)
        return -NLL.sum()

    def evaluate_linear(self, slope):
        """
        Evaluate negative log likelihood for gaussian signal and linear background
        """

        #This is where the params def takes effect as we can now set different parameters as we want for the minimisation
        self.pdf.params(slope = slope)
        norm = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood = self.pdf.evaluate(self.data,)/norm 
        # fudge factor for negative or 0 values. These are set very close to 0
        if (evaluate_likelihood <= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5
        NLL = np.log(evaluate_likelihood)
        return -NLL.sum()

    def evaluate_flat(self, y_int):
        """
        Evaluate negative log likelihood for gaussian signal and linear background
        """

        #This is where the params def takes effect as we can now set different parameters as we want for the minimisation
        self.pdf.params(y_int = y_int)
        norm = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood = self.pdf.evaluate(self.data,)/norm 
        # fudge factor for negative or 0 values. These are set very close to 0
        if (evaluate_likelihood <= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5
        NLL = np.log(evaluate_likelihood)
        return -NLL.sum()

    def evaluate_polynomial(self,  a, b, c):
        """
        Evaluate negative log likelihood for gaussian signal and linear background
        """

        #This is where the params def takes effect as we can now set different parameters as we want for the minimisation
        self.pdf.params( a = a, b = b, c = c)
        norm = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood = self.pdf.evaluate(self.data,)/norm 
        # fudge factor for negative or 0 values. These are set very close to 0
        if (evaluate_likelihood <= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5
        NLL = np.log(evaluate_likelihood)
        return -NLL.sum()

    def evaluate_harmonicdecay(self, tau, delta_mass, V):
        """
        computes and returns the negative log likelihood of the pdf and data
        """
        # first set the new parameters passed. This is where the params definition in the above class
        # comes into effect
        self.pdf.params(tau=tau, delta_mass=delta_mass, V=V)

        #  compute the normalisation factor and the likelihood
        normalisation = self.pdf.integrate((self.pdf.XMIN, self.pdf.XMAX))
        evaluate_likelihood= self.pdf.evaluate(self.data,) / normalisation
        
        # the fudge factor for values <=0. set them to a very small value close to 0
        if (evaluate_likelihood<= 0).any():
            evaluate_likelihood[evaluate_likelihood<=0 ] = 1e-5

        # compute log and return thr negative sum
        return -(np.log(evaluate_likelihood).sum())
