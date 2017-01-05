import numpy as np
from sgld import StochasticGradientLangevinDynamics


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


    def __init__(self,X_train,X_test,y_train,y_test):
        """
        Initialise the logistic regression object.

        Parameters:
        X_train - matrix of explanatory variables for training (assumes numpy array of floats)
        X_test - matrix of explanatory variables for testing (assumes numpy array of ints)
        y_train - vector of response variables for training (assumes numpy array of ints)
        y_train - vector of response variables for testing (assumes numpy array of ints)
        """
        # Set error to be raised if there's an over/under flow
        np.seterr( over = 'raise', under = 'raise' )
        self.X = X_train
        self.y = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Set dimension constants
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.test_size = self.X_test.shape[0]
        
        # Initialise parameters
        self.beta = np.zeros(self.d)
        self.sample = None
        self.training_loss = []


    def fit(self,stepsize,n_iters=10**4,minibatch_size=500):
        """
        Fit Bayesian logistic regression model using train and test set.

        Uses stochastic gradient Langevin dynamics algorithm

        Parameters:
        stepsize - stepsize to use in stochastic gradient descent
        n_iters - number of iterations of stochastic gradient descent (optional)
        minibatch_size - minibatch size in stochastic gradient descent (optional)
        """
        # Holds log loss values once fitted
        self.training_loss = []
        # Number of iterations before the logloss is stored
        self.loss_thinning = 100
        # Initialize sample storage
        self.sample = np.zeros( ( n_iters, self.d ) )

        sgld = StochasticGradientLangevinDynamics(self,stepsize,minibatch_size,n_iters)
        print "{0}\t{1}".format( "iteration", "Test log loss" )
        for sgld.iter in range(1,n_iters+1):
            # Every so often output log loss on test set and store 
            if sgld.iter % self.loss_thinning == 0:
                current_loss = self.logloss()
                self.training_loss.append( current_loss )
                print "{0}\t\t{1}".format( sgld.iter, current_loss )
            sgld.update(self)
            self.sample[(sgld.iter-1),:] = self.beta


    def logloss(self):
        """Calculate the log loss on the test set, used to check convergence"""
        logloss = 0
        for i in range(self.test_size):
            y = self.y_test[i]
            x = np.squeeze( np.copy( self.X_test[i,:] ) )
            p = 1 / ( 1 + np.exp( - np.dot( self.beta, x ) ) )
            logloss -= 1 / float(self.test_size) * ( y * np.log( p ) + ( 1 - y ) * np.log( 1 - p ) )
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
        for i in sgld.minibatch:
            x = np.squeeze( np.copy( self.X[i,:] ) )
            y = self.y[i]
            # Calculate gradient of the log density at current point, use to update dlogbeta
            # Handle overflow gracefully by catching numpy's error
            # (seterr was defined at start of class)
            
            dlogbeta += ( y - 1 / ( 1 + np.exp( - np.dot( self.beta, x ) ) ) ) * x

        # Adjust log density gradients so they're unbiased
        dlogbeta *= self.N / sgld.minibatch_size
        # Add gradient of log prior (assume Laplace prior with scale 1)
        dlogbeta -= np.sign(self.beta)
        return dlogbeta
