from synthetic import Synthetic
import GPy
import GPyOpt
import numpy as np


def b_opt( init_n = 4, n_its = 6 ):
    example = Synthetic( seed = 13 )
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,0.01)}]  # problem constrains
    # Generate initial points
    body_odour = GPyOpt.methods.BayesianOptimization(f = example.log_loss_objective,
            domain = bounds,
            acquisition_type ='EI' )
    body_odour.run_optimization( max_iter = 10 )
    print body_odour.model.model
    print body_odour.get_evaluations()
    body_odour.plot_acquisition()

if __name__ == '__main__':
    b_opt()
