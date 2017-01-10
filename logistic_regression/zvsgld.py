import numpy as np
import pkg_resources


class ZVSGLD:
    """
    Methods to apply SGLD with zero variance control variate postprocessing for logistic regression

    SGLD stands for stochastic gradient Langevin dynamics and is a MCMC method for large datasets.
    Zero variance control variates are used to improve the efficiency of the sample.

    SGLD notation used as in reference 1
    Zero variance control variate notation used as in reference 2
    References:
        1. Stochastic gradient Langevin dynamics - 
                http://people.ee.duke.edu/~lcarin/398_icmlpaper.pdf
        2. Zero variance control variates for Hamiltonian Monte Carlo - 
                https://projecteuclid.org/download/pdfview_1/euclid.ba/1393251772
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
        # TEMPORARILY SAVED FILES FOR BUILDING, REMOVE!!!
        data_dir = pkg_resources.resource_filename('logistic_control_variate', 'data/')
        lr.sample = np.load( data_dir + 'temp_postprocess/sample.pkl' )
        lr.grad_sample = np.load( data_dir + 'temp_postprocess/grad_sample.pkl' )
        ####
        # discard burn in
        # Calculate key variables for calculation
        converged_sample = lr.sample[1000:]
        converged_grad = lr.sample[1000:]
        sample_mean = np.mean( converged_sample, axis = 0 )
        grad_mean = np.mean( converged_grad, axis = 0 )
        M, D = converged_sample.shape
        var_grad = np.cov( converged_grad, rowvar = 0 )

        # Initialise variables
        out_sample = np.zeros( ( M, D ) )
        a = np.zeros(D)
        current_cov = np.zeros(D)

        # Calculate new sample once control variates have been calculated
        for j in range(D):
            current_cov = np.zeros(D)
            for i in range(M):
                current_cov += 1 / float(M-1) * ( converged_sample[i,j] - sample_mean[j] ) * ( 
                        converged_grad[i,:] - grad_mean )
            a = - np.matmul( np.linalg.inv( var_grad ), current_cov )
            for i in range(M):
                out_sample[i,j] = converged_sample[i,j] + np.dot( a, converged_grad[i,:] )

        # Calculate new log loss at a subsample of points
        subsample = np.random.choice( range(M), 10 )
        for i in subsample:
            oldll = lr.loglossp(converged_sample[i,:])
            newll = lr.loglossp(out_sample[i,:])
            print "Old log loss: {0}\tNew log loss: {1}".format( oldll, newll )

    def sample_minibatch(self,lr):
        """Sample the next minibatch"""
        self.minibatch = np.random.choice( np.arange( lr.N ), self.minibatch_size, replace = False )
