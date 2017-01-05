import numpy as np


class StochasticGradientLangevinDynamics:
    """
    Methods to apply stochastic gradient Langevin dynamics updates for logistic regression

    Notation used as in reference 1
    References:
        1. http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
    """
    
    def __init__(self,lr,epsilon,minibatch_size,n_iter):
        """
        Initialize the container for SGLD

        Parameters:
        lr - LogisticRegression object
        epsilon - the stepsize to perform SGD at
        minibatch_size - size of the minibatch used at each iteration
        n_iter - the number of iterations to perform
        """
        self.epsilon = epsilon
        # Set the minibatch size
        self.minibatch_size = minibatch_size
        self.sample_minibatch(lr)
        # Hold number of iterations so far
        self.iter = 1
        self.output = np.zeros( ( n_iter, lr.d ) )


    def update(self,lr):
        """
        Update one step of stochastic gradient Langevin dynamics

        Parameters:
        lr - LogisticRegression object
        """
        self.sample_minibatch(lr)
        # Calculate gradients at current point
        dlogbeta = lr.dlogpost(self)

        # Update parameters using SGD
        eta = np.random.normal( scale = self.epsilon )
        lr.beta += self.epsilon / 2 * dlogbeta + eta


    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
