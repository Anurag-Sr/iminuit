import numpy as np
from scipy import integrate


class PDF(object):
    """
    Parent class pdf object. Implemenst the integrate and Mass definitions.
    This is  stictly necessary but was part of me exploring classes in
    python
    """

    def __init__(self, XRange):
        """
        initialises limits and mass for this class
        """

        self.XMIN, self.XMAX = XRange
        self.mass = []

    def integrate(self, XRange):
        """
        uses scipy.integrate.quad to evaluate and returns the integral value
        """

        XMIN, XMAX = XRange
        integral, _ = integrate.quad(self.evaluate, XMIN, XMAX)
        return integral

    def Mass(self,):
        """
        Return numpy array containing all generated values
        """

        return np.array(self.mass)


class Gaussian(PDF):
    """
    class to generate random signal with gaussian shape
    """

    def __init__(self, mean, sigma, cut=0.0, XRange=(0, 10)):

        super().__init__(XRange)

        self.mean = mean
        self.sigma = sigma
        self.cut = cut
        self.mass = []
        self.XMIN, self.XMAX = XRange
        # Find maximum value of the gaussian function
        self.max = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def evaluate(self, x):
        """
        evaluates the gaussian function e this is normalised
        """

        return np.exp(-(x - self.mean)**2 / (2.0 * self.sigma**2))

    def next(self):
        """
        generates gaussian values of x for signal using np.random.normal
        """

        x = np.random.normal(self.mean, self.sigma, size=1).item()
        self.mass.append(x)
        return x

    def params(self, mean=None, sigma=None):
        """
        Sets parameters passed. Mainly used for negative loglikelihood statistics where we want to vary the parameters.
        """
        if mean is not None:
            self.mean = mean
        if sigma is not None:
            self.sigma = sigma


class Exponential(PDF):
    """
    Class to generate an object with an exponential decay (for background)
    """

    def __init__(self, decay_constant, XRange=(0, 10)):
        super().__init__(XRange)

        self.decay_constant = decay_constant
        self.mass = []
        self.XMIN, self.XMAX = XRange
        self.max = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def next(self,):
        """
        generates points according to the required background fraction. This uses the numpy.random.exponential method
        """

        x = np.random.exponential(self.decay_constant)
        self.mass.append(x)
        return x

    def evaluate(self, x):
        """
        evaluates exponential function. This is  normalised to the range
        """

        return np.exp(-x / self.decay_constant)

    def params(self, decay_constant=None):
        """
        Sets parameters passed. Mainly used for negative log ikelihood statistics where we want to vary the parameters.
        """

        if decay_constant is not None:
            self.decay_constant = decay_constant


class Linear(PDF):
    """
    Class for the linear function PDF
    """

    def __init__(self, slope, XRange=(0, 10)):

        super().__init__(XRange)

        self.slope = slope
        self.mass = []
        self.XMIN, self.XMAX = XRange
        # maximum of a linear function
        self.max = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def evaluate(self, x):
        """
        evaluates the linear function. This is normalised to the checkpoint
        """

        return (1.0 + self.slope * x)

    def next(self):
        """
        Function to draw random x values
        """

        while True:
            # Generate uniform random numbers within limits
            x = np.random.uniform(self.XMIN, self.XMAX)
            # evaluate corresponding y value for each x
            y1 = self.evaluate(x)
            # generate random uniform distribution of y values with max y of function
            y2 = np.random.uniform(0, self.max)
            # accept x value if y1<y2
            if (y2 < y1):
                filtered_x = x
                self.mass.append(filtered_x)
                return filtered_x

    def params(self, slope=None):
        """
        to set passed parameters
        """
        if slope is not None:
            self.slope = slope


class Polynomial(PDF):
    """
    seecond order polynomial class
    """

    def __init__(self, a, b, c, XRange=(0, 10)):

        super().__init__(XRange)

        # Initialise class variables
        self.a = a
        self.b = b
        self.c = c
        self.mass = []
        self.XMIN, self.XMAX = XRange
        # Find maximum value of the gaussian function
        self.max = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def evaluate(self, x,):
        """
        Evaluates the quadratic polynomial.
        """

        return (self.a + self.b * x + self.c * x**2)

    def next(self):
        """
        Function to draw random x values
        """

        while True:
            # Generate uniform random numbers within limits
            x = np.random.uniform(self.XMIN, self.XMAX)
            # evaluate corresponding y value for each x
            y1 = self.evaluate(x)
            # generate random uniform distribution of y values with max y of function
            y2 = np.random.uniform(0, self.max)
            # accept x value if y1<y2
            if (y2 < y1):
                filtered_x = x
                self.mass.append(filtered_x)
                return filtered_x

    def params(self, a=None, b=None, c=None):
        """
        Sets parameters passed. Mainly used for negative loglikelihood statistics where we want to vary the parameters.
        """

        if a is not None:
            self.a = a
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c


class Flat(PDF):

    """
    generates events for background flat function using np.random.uniform
    (modified copy of linear function from week 8)
    """

    def __init__(self, y_int, XRange=(0, 10)):

        super().__init__(XRange)

        self.y_int = y_int
        self.XMIN, self.XMAX = XRange
        self.mass = []
        # Find maximum value of the distribution within the bounds
        self.max_val = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def evaluate(self, x,):
        """
        evaluates flat function
        """

        return np.array(self.y_int)

    def next(self,):
        """
        uses same principle as described in week 8 to generate events
        unlike gaussian this uses np.random.uniform
        """

        while True:
            x = np.random.uniform(self.XMIN, self.XMAX)
            y1 = self.evaluate(x)
            y2 = np.random.uniform(0, self.max_val)
            if (y2 < y1):
                filtered_x = x
                self.mass.append(filtered_x)
                return filtered_x

    def params(self, y_int=None):
        """
        Sets parameters passed. Mainly used for negative loglikelihood statistics where we want to vary the parameters.
        """

        if y_int is not None:
            self.y_int = y_int


class harmonic_decay(PDF):
    """
    Class that would generate an PDF with a randomly generated dataset of n points according to the
    harmonic decay function
    """

    def __init__(self, tau, delta_mass, V, XRange=(0, 10)):

        # Initialises all the variables used in the class, range is set by default to (0,10)

        super().__init__(XRange)

        self.XMIN, self.XMAX = XRange = (0, 10)
        self.tau = tau
        self.delta_mass = delta_mass
        self.V = V
        self.mass = []

        # find maximum of the function in the range provided
        self.max = self.find_maximum()

    def find_maximum(self,):
        """
        Returns maximum value of the function
        """
        # create a grid of 10000 points that will be used to compute and return the maximum
        x = np.linspace(self.XMIN, self.XMAX, num=10000, endpoint=True,)
        y = self.evaluate(x)
        return y.max()

    def evaluate(self, t,):
        """
        Evaluates the harmonic decay function. This function is used in integrate and next to execute
        the necessary steps. e this is  normalised
        """

        return (1 + self.V * np.sin(self.delta_mass * t)) * np.exp(-t / self.tau)

    def next(self,):
        """
        This function generates a set of n points and appends them to the mass list. The box method is used to generate the points.
        """
        while True:
            x = np.random.uniform(self.XMIN, self.XMAX)
            y = self.evaluate(x,)
            y_points = np.random.uniform(self.XMIN, self.max, )

            # accept points based of the conditions of the box method.
            if (y_points <= y):
                self.mass.append(x)
                return (x, y_points)

    def params(self, tau=None, delta_mass=None, V=None):
        """
        This function is used to set the parameters passed through the PDF.
        Mainly used during negative log likelihood fitting where we try to
        fit the data by varrying 1 or more parameters.

        If the parameter is  passed it is set to the default value
        """

        if tau is not None:
            self.tau = tau
        if delta_mass is not None:
            self.delta_mass = delta_mass
        if V is not None:
            self.V = V
