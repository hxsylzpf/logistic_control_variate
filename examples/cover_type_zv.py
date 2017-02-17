import os
import sys
import pickle
import pkg_resources
import urllib
import numpy as np
import bz2
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from ..logistic_regression.logistic_regression import LogisticRegression


class CoverType:
    """
    Example for fitting a logistic regression model to the cover type dataset

    References:
    1. Cover type dataset - https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
    """

    def __init__(self):
        """Load data into the object"""
        self.data_dir = pkg_resources.resource_filename('logistic_control_variate', 'data/')
        self.lr = None
        # Try opening the data, if it's not available, download it.
        try:
            np.load( self.data_dir + 'cover_type/X_train.dat' )
        except IOError:
            raw = self.download_data()
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()
            self.X_train.dump( self.data_dir + 'cover_type/X_train.dat' )
            self.X_test.dump( self.data_dir + 'cover_type/X_test.dat' )
            self.y_train.dump( self.data_dir + 'cover_type/y_train.dat' )
            self.y_test.dump( self.data_dir + 'cover_type/y_test.dat' )
        else:
            self.X_train = np.load( self.data_dir + 'cover_type/X_train.dat' )
            self.X_test = np.load( self.data_dir + 'cover_type/X_test.dat' )
            self.y_train = np.load( self.data_dir + 'cover_type/y_train.dat' )
            self.y_test = np.load( self.data_dir + 'cover_type/y_test.dat' )
    
    
    def truncate(self,train_size,test_size):
        self.X_train = self.X_train[:train_size,:]
        self.y_train = self.y_train[:train_size]
        self.X_test = self.X_test[:test_size,:]
        self.y_test = self.y_test[:test_size]


    def fit(self,stepsize):
        """
        Fit a Bayesian logistic regression model to the data using the LogisticRegression class.

        Parameters:
        stepsize - stepsize parameter for the stochastic gradient langevin dynamics

        Returns:
        lr - fitted LogisticRegression object
        """
        self.lr = LogisticRegression( self.X_train, self.X_test, self.y_train, self.y_test )
        self.lr.fit(stepsize, n_iters = 10**4)


    def download_data(self):
        """Download raw cover type data"""
        if not os.path.exists( self.data_dir + 'cover_type' ):
            os.makedirs( self.data_dir + 'cover_type' )
        print "Downloading data..."
        urllib.urlretrieve( ( "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
                "datasets/binary/covtype.libsvm.binary.scale.bz2" ), 
                self.data_dir + "cover_type/covtype.libsvm.binary.bz2" )
        compressed_file = bz2.BZ2File( self.data_dir + 'cover_type/covtype.libsvm.binary.bz2', 
                'r')
        with open( self.data_dir + 'cover_type/raw.dat', 'w' ) as out: 
            out.write( compressed_file.read() )


    def preprocess(self):
        """Preprocess raw data once downloaded, split into train and test sets"""
        X, y = load_svmlight_file( self.data_dir + 'cover_type/raw.dat' )
        X_train, X_test, y_train, y_test = train_test_split( X, y  )
        # Set y to go from 0 to 1
        y_train -= 1
        y_test -= 1
        # Add a bias term to X
        X_train = np.concatenate( ( np.ones( ( len(y_train), 1 ) ), X_train.todense() ), axis = 1 )
        X_test = np.concatenate( ( np.ones( ( len(y_test), 1 ) ), X_test.todense() ), axis = 1 )
        return X_train, X_test, y_train.astype(int), y_test.astype(int)


    def simulation(self,stepsize,current_seed):
        self.fit(stepsize)
        if not os.path.exists( self.data_dir + 'simulations/{0}/{1}'.format(stepsize,current_seed) ):
            os.makedirs( self.data_dir + 'simulations/{0}/{1}'.format(stepsize,current_seed) )
        llold, llnew = self.lr.postprocess()
        print( np.mean( llold ) )
        print( np.mean( llnew ) )
        print( np.cov( llold ) )
        print( np.cov( llnew ) )
        np.savetxt( self.data_dir + 'simulations/{0}/{1}/old.dat'.format(stepsize,current_seed), llold )
        np.savetxt( self.data_dir + 'simulations/{0}/{1}/new.dat'.format(stepsize,current_seed), llnew )

    
    def postprocess(self):
        llold, llnew = self.lr.postprocess()
        print( np.mean( llold ) )
        print( np.mean( llnew ) )
        print( np.cov( llold ) )
        print( np.cov( llnew ) )
        np.savetxt( self.data_dir + 'simulations/{0}/{1}/old.dat'.format(stepsize,current_seed), llold )
        np.savetxt( self.data_dir + 'simulations/{0}/{1}/new.dat'.format(stepsize,current_seed), llnew )


if __name__ == '__main__':
    print "Simulation started!"
    example = CoverType()
    stepsizes = [1e-6, 1e-5, 3e-5, 7e-5, 1e-4] 
    step_index = int( sys.argv[1] ) - 1
    stepsize = stepsizes[step_index]
    current_seed = sys.argv[2]
    seed(current_seed)
    example.simulation(stepsize,current_seed)
#    with open( example.data_dir + 'simulations/{0}/{1}/lr.pkl'.format(stepsize,current_seed), 'w' ) as outfile:
#        pickle.dump( example, outfile )
#    with open( example.data_dir + 'simulations/{0}/{1}/lr.pkl'.format(stepsize,current_seed), 'r' ) as infile:
#        example = pickle.load( infile )
#    example.postprocess()
