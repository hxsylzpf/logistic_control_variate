import unittest
import sys
import os
import random
import numpy as np
from scipy.stats import f
from ..examples.synthetic import Synthetic

class SGLDTests( unittest.TestCase ):
    """Tests for training a logistic regression model using SGLD, uses synthetic random data"""


    def testSimilar(self):
        """Check that the same model fits to similar beta values under different random seeds"""
        # Initialise two synthetic models with the same seed
        synthetic_test1 = Synthetic( seed = 5 )
        synthetic_test2 = Synthetic( seed = 5 )
        # Silence output of fit
        save_stdout = sys.stdout
        sys.stdout = open( os.devnull, 'w' )
        # Fit models under two different seeds
        random.seed(10)
        synthetic_test1.fit(0.0005, n_iters = 10**3)
        random.seed(13)
        synthetic_test2.fit(0.0005, n_iters = 10**3)
        sys.stdout.close()
        sys.stdout = save_stdout

        # Check for closeness using Hotelling T^2 test (test for 0 mean vector)
        diffs = synthetic_test1.lr.sample - synthetic_test2.lr.sample
        # Discard burn-in
        diffs = diffs[100:]
        Xbar = np.mean( diffs, axis = 0 )
        S = np.cov( diffs, rowvar = 0 )
        n = diffs.shape[0]
        d = diffs.shape[1]
        Tsq = np.dot( Xbar, np.matmul( np.linalg.inv( S ), Xbar ) ) / float( diffs.shape[0] )
        F = ( n - d ) / float( d * ( n - 1 ) ) * Tsq
        alpha = 0.95
        critical_value = f.ppf( 1 - alpha / 2.0, d, n-d )
        self.assertTrue( F < critical_value )


    def testConvergence(self):
        """
        Test for algorithm convergence on a random synthetic example once the model is fit
        
        Test checks that the variance in the log loss is small
        """
        synthetic_test = Synthetic()
        # Silence output of fit
        save_stdout = sys.stdout
        sys.stdout = open( os.devnull, 'w' )
        synthetic_test.fit(0.001, n_iters = 10**3)
        sys.stdout.close()
        sys.stdout = save_stdout
        # Discard burn in
        loss_storage = np.array( synthetic_test.lr.training_loss[1:] )
        loss_var = np.var( loss_storage )
        self.assertTrue( loss_var < 1 / float( synthetic_test.lr.N ) )


if __name__ == '__main__':
    unittest.main()
