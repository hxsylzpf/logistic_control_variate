import numpy as np
import pkg_resources
from sklearn.metrics import log_loss


class CVSGLD:
    """
    Methods to apply SGLD with control variates to reduce gradient noise for logistic regression

    SGLD stands for stochastic gradient Langevin dynamics and is a MCMC method for large datasets.
    The control variates are used to reduce the variance in the log posterior gradient estimate.

    SGLD notation used as in reference 1
    References:
        1. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
        2. Control variates for big data MCMC
                https://arxiv.org/abs/1607.03188    
    """
    
    def __init__(self,lr,epsilon,minibatch_size,n_iter):
        """
        Initialize the container for CVSGLD

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

        Modifies:
        lr.beta - updates parameter values using SGLD
        lr.grad_sample - adds calculated gradient to storage
        """
        self.sample_minibatch(lr)
        # Calculate gradients at current point
        dlogbeta = lr.dlogpost(self)
        lr.grad_sample[self.iter-1,:] = dlogbeta

        # Update parameters using SGD
        eta = np.random.normal( scale = self.epsilon )
        lr.beta += self.epsilon / 2 * dlogbeta + eta


    def control_variates(self,lr):
        """
        Postprocess a fitted LogisticRegression object using zero variance control variates.

        Assumes object has already been fitted using SLGD i.e. lr.sample is nonempty.

        Parameters:
        lr - fitted LogisticRegression object

        Modifies:
        lr.sample - updates stored MCMC chain using ZV control variates
        """
        pot_energy = - 1 / 2.0 * lr.grad_sample
        sample_mean = np.mean( lr.sample, axis = 0 )
        grad_mean = np.mean( pot_energy, axis = 0 )
        var_grad_inv = np.linalg.inv( np.cov( pot_energy, rowvar = 0 ) )

        # Initialise variables
        cov_params = np.zeros( lr.d )
        a_current = np.zeros( lr.d )
        new_sample = np.zeros( lr.sample.shape )

        # Calculate covariance for each parameter
        for j in range(lr.d):
            cov_params = np.zeros(lr.d)
            a_current = np.zeros( lr.d )
            for i in range(self.n_iters):
                cov_params += 1 / float( self.n_iters - 1 ) * ( 
                        lr.sample[i,j] - sample_mean[j] ) * ( pot_energy[i,j] - grad_mean[j] )
            # Update sample for current dimension
            a_current = - np.matmul( var_grad_inv, cov_params )
            for i in range(self.n_iters):
                new_sample[i,j] = lr.sample[i,j] + np.dot( a_current, pot_energy[i,:] )
        # Compare new samples
        print np.cov( lr.sample )
        print np.cov( new_sample )


                



    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
