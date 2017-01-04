import numpy as np
from sgd import StochasticGradientDescent


class LogisticRegression:
    """
    Methods for performing Bayesian logistic regression for large datasets.

    Logistic regression is trained using stochastic gradient Langevin dynamics
    with control variate postprocessing.

    References: 
        1. Logistic regression - https://www.stat.cmu.edu/~cshalizi/uADA/12/lectures/ch12.pdf
        2. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
        3. Control variates for MCMC -
                https://projecteuclid.org/download/pdfview_1/euclid.ba/1393251772
    """


    def __init__(X_train,X_test,y_train,y_test)
        """
        Initialise the logistic regression object.

        Parameters:
        X_train - matrix of explanatory variables for training (assumes numpy array of floats)
        X_test - matrix of explanatory variables for testing (assumes numpy array of floats)
        y_train - vector of response variables for training (assumes numpy array of floats)
        y_train - vector of response variables for testing (assumes numpy array of floats)
        """
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set dimension constants
        self.N = self.y.shape
        self.d = self.X.shape[1]
        self.test_size = y_test.shape
        
        # Initialise parameters
        self.beta = np.zeros(self.d)


    def fit(self,stepsize,n_iters=10**4,minibatch_size=4000):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        sgd = StochasticGradientLangevinDynamics(self,stepsize,minibatch_size)
        print "{0}\t{1}".format( "iteration", "Test log loss" )
        for i in range(n_iters):
            # Every so often output log loss on test set and progress
            if i % 10 == 0:
                print "{0}\t\t{1}".format( i, self.rmse() )
            sgd.update(self)
            sgd.iter += 1


    def logloss(self):
        """Calculate the log loss on the test set, used to check convergence"""
        logloss = 0
        for i in range(self.test_size):
            y = self.y_test[i]
            p = 1 / ( 1 + np.dot( self.beta, self.X_test[i,:] ) )
            logloss -= 1 / self.test_size * ( y * np.log( p ) + ( 1 - y ) * np.log( 1 - p ) )
        return logloss


    def dlogpost(self,sgld):
        """
        Calculate gradient of the log posterior wrt the parameters using a minibatch of data

        Parameters:
        sgld - a StochasticGradientLangevinDynamics object, used to specify the minibatch

        Returns:
        dlogbeta - gradient of the log likelihood wrt the parameter beta 
        """
        dlogbeta = np.zeros( self.d )

        # Calculate sum of gradients at each point in the minibatch
        for i in sgd.minibatch:
            x = self.X[i,:]
            y = self.y[i]
            # Calculate gradient of the log density at current point, use to update dlogbeta
            dlogbeta += 1 / ( 1 + np.exp( y * np.dot( self.beta, x ) ) ) * y * x

        # Adjust log density gradients so they're unbiased
        dlogbeta *= self.N / sgd.minibatch_size
        # Add gradient of log prior (assume Laplace prior with scale 1)
        dlogbeta -= np.sign(self.beta)
        return dlogbeta
