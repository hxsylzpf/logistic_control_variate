import os
import pkg_resources
import pickle
import urllib
import numpy as np
import bz2
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ..hmc.lrhmc import LRHMC


class Synthetic:
    """Example that fits the logistic regression model to a synthetic dataset for simple testing"""

    def __init__(self, seed = None):
        """Load data into the object"""
        self.data_dir = pkg_resources.resource_filename('logistic_control_variate', 'data/')
        self.generate_data(seed)
        # Holds logistic regression object for this example
        self.lr = None


    def fit(self):
        """
        Fit a Bayesian logistic regression model to the data using the LogisticRegression class.

        Parameters:
        stepsize - stepsize parameter for the stochastic gradient langevin dynamics

        Returns:
        lr - fitted LogisticRegression object
        """
        self.lr = LRHMC( self.X_train, self.X_test, self.y_train, self.y_test )
        self.lr.fit()


    def generate_data(self,seed):
        """Generate synthetic dataset using standard methods in scikit-learn"""
        X, y = make_classification( n_samples = 250, random_state = seed )
        # Add bias term
        X = np.concatenate( ( np.ones( ( 250, 1 ) ), X ), axis = 1 )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split( 
                X, y, test_size = 50, random_state = seed )


if __name__ == '__main__':
    filename = pkg_resources.resource_filename(
            'logistic_control_variate','data/hmc_temp/fitted.pkl' )
#    with open( filename ) as hmc_in:
#        lr = pickle.load( hmc_in )
#    lr.postprocess()
    example = Synthetic( seed = 13 )
    lr = example.fit()
