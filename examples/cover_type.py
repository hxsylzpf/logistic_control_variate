import os
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
        # Try opening the data, if it's not available, download it.
        try:
            np.load( self.data_dir + 'cover_type/X_train.dat' )
        except IOError:
            raw = self.download_data()
            self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess()
            X_train.dump( self.data_dir + 'cover_type/X_train.dat' )
            X_test.dump( self.data_dir + 'cover_type/X_test.dat' )
            y_train.dump( self.data_dir + 'cover_type/y_train.dat' )
            y_test.dump( self.data_dir + 'cover_type/y_test.dat' )
        else:
            self.X_train = np.load( self.data_dir + 'cover_type/X_train.dat' )
            self.X_test = np.load( self.data_dir + 'cover_type/X_test.dat' )
            self.y_train = np.load( self.data_dir + 'cover_type/y_train.dat' )
            self.y_test = np.load( self.data_dir + 'cover_type/y_test.dat' )


    def fit(self,stepsize):
        """
        Fit a Bayesian logistic regression model to the data using the LogisticRegression class.

        Parameters:
        stepsize - stepsize parameter for the stochastic gradient langevin dynamics

        Returns:
        lr - fitted LogisticRegression object
        """
        lr = LogisticRegression( self.X_train, self.X_test, self.y_train, self.y_test )
        lr.fit(stepsize)


    def download_data(self):
        """Download raw cover type data"""
        if not os.path.exists( self.data_dir + 'cover_type' ):
            os.makedirs( self.data_dir + 'cover_type' )
        print "Downloading data..."
        urllib.urlretrieve( ( "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/"
                "datasets/binary/covtype.libsvm.binary.bz2" ), 
                self.data_dir + "cover_type/covtype.libsvm.binary.bz2" )
        compressed_file = bz2.BZ2File( self.data_dir + 'cover_type/covtype.libsvm.binary.bz2', 
                'r')
        with open( self.data_dir + 'cover_type/raw.dat', 'w' ) as out: 
            out.write( compressed_file.read() )


    def preprocess(self):
        """Preprocess raw data once downloaded, split into train and test sets"""
        X, y = load_svmlight_file( self.data_dir + 'cover_type/raw.dat' )
        X_train, X_test, y_train, y_test = train_test_split( X, y  )
        y_train -= 1
        y_test -= 1
        return X_train.todense(), X_test.todense(), y_train.astype(int), y_test.astype(int)


if __name__ == '__main__':
    example = CoverType()
    example.fit(1e-5)
