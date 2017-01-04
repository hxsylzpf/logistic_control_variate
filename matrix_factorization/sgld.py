import numpy as np


class StochasticGradientLangevinDynamics:
    """
    Container which holds data necessary for a stochastic gradient Langevin dynamics update 
    for logistic regression

    Notation used as in reference 1
    References:
        1. http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
    """
    
    def __init__(self,lr,epsilon,minibatch_size):
        """
        Initialize the container for SGLD

        Parameters:
        lr - LogisticRegression object
        epsilon - the stepsize to perform SGD at
        minibatch_size - size of the minibatch used at each iteration
        """
        self.epsilon = epsilon

        # Set the minibatch size
        self.minibatch_size = minibatch_size
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
        
        # Hold number of iterations so far
        self.iter = 1


    def update(self,lr):
        """
        Update one step of stochastic gradient Langevin dynamics

        Parameters:
        lr - LogisticRegression object
        """
        # Sample the next minibatch
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )

        # Calculate gradients at current point
        dlogbeta = pmf.dlogpost(self)

        # Update parameters using SGD
        eta = np.random.normal( scale = self.epsilon )
        lr.beta += self.stepsize / 2 * dlogbeta + eta
