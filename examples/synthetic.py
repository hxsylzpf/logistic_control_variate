import os
import pkg_resources
import pickle
import urllib
import numpy as np
import bz2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ..logistic_regression.logistic_regression import LogisticRegression


class Synthetic:
    """Example that fits the logistic regression model to a synthetic dataset for simple testing"""

    def __init__(self, seed = None):
        """Load data into the object"""
        self.data_dir = pkg_resources.resource_filename('logistic_control_variate', 'data/')
        self.generate_data(seed)
        # Holds logistic regression object for this example
        self.lr = None


    def fit(self,stepsize, n_iters = 10**4):
        """
        Fit a Bayesian logistic regression model to the data using the LogisticRegression class.

        Parameters:
        stepsize - stepsize parameter for the stochastic gradient langevin dynamics

        Returns:
        lr - fitted LogisticRegression object
        """
        self.lr = LogisticRegression( self.X_train, self.X_test, self.y_train, self.y_test )
        self.lr.fit(stepsize, n_iters = n_iters)


    def generate_data(self,seed):
        """Generate synthetic dataset using standard methods in scikit-learn"""
        X, y = make_classification( n_samples = 7000, random_state = seed )
        # Add bias term
        X = np.concatenate( ( np.ones( ( 7000, 1 ) ), X ), axis = 1 )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( 
                X, y, test_size = 2000, random_state = seed )


    def log_loss_objective( self, stepsize_mat ):
        n_rows = stepsize_mat.shape[0]
        outputs = np.zeros( stepsize_mat.shape )
        for i in range(n_rows):
            stepsize_curr = stepsize_mat[i,0]
            try:
                self.fit(stepsize_curr,10**3)
            except FloatingPointError:
                outputs[i,0] = 8.0
            outputs[i,0] = np.array( self.lr.training_loss ).mean()
        return outputs


if __name__ == '__main__':
    print log_loss_objective( 0.05, 13 )
