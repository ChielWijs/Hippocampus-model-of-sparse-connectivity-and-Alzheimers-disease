import logging
import time

import numpy as np

import nengo.utils.least_squares_solvers as lstsq
from nengo.exceptions import ValidationError
from nengo.params import (
    BoolParam, FrozenObject, NdarrayParam, NumberParam, Parameter)
from nengo.utils.compat import range, with_metaclass
from nengo.utils.least_squares_solvers import (
    format_system, rmses, LeastSquaresSolverParam)
from nengo.utils.magic import DocstringInheritor
from nengo.solvers import *
import random

logger = logging.getLogger(__name__)

class LstsqSparseWeights(Solver):
    """Find sparser decoders/weights by dropping values.


    Based on the LstsqDrop solver, this solver solves for 
    coefficients (decoders/weights) with L2 regularization, 
    drops those lowest/highest/random according to the drop 
    percentage, but does NOT retrain the non-zero weights (as opposed to LstsqDrop).
    """

    drop = NumberParam('drop', low=0, high=1)
    solver = SolverParam('solver')

    def __init__(self, drop=0.0, drop_method="lowest", excitatory=False,
                 solver=LstsqL2(reg=0.1), seed=None):
        """
        Parameters
        ----------
        drop : float, optional (Default: 0.0)
            Fraction of weights to set to zero.
        drop_method : String (Default: "lowest")
            Method used for dropping weights. Use: "lowest", 
            "highest", or "random".
        excitatory : Bool (Default: False)
            Indicates whether als solved weights are converted to 
            positive weights.
        solver : Solver, optional (Default: ``LstsqL2(reg=0.1)``)
            Solver for finding the initial weights.

        Attributes
        ----------
        drop : float
            Fraction of weights to set to zero.
        drop_method : String
            Method used for dropping weights. Use "lowest", 
            "highest", or "random".
        excitatory : Bool (Default: False)
            Indicates whether als solved weights are converted to 
            positive weights.
        solver : Solver
            Solver for finding the initial weights.
        """
        super(LstsqSparseWeights, self).__init__(weights=True)
        self.drop = drop
        self.solver = solver
        self.drop_method = drop_method
        self.excitatory = excitatory
        self.seed = seed
        
        if self.drop_method not in ["lowest", "highest", "random"]:
            error("Please specify ``drop_method'' as 'lowest', 'highest', or 'random'.")

    def __call__(self, A, Y, rng=np.random, E=None):
        tstart = time.time()
        Y, _, _, _, matrix_in = format_system(A, Y)

        # solve for coefficients using standard solver
        X, _ = self.solver(A, Y, rng=rng)
        X = self.mul_encoders(X, E)

        if self.drop != 0: 

            if self.drop_method == "lowest":

                Xabs = np.sort(np.abs(X.flat))
                threshold = Xabs[int(np.round(self.drop * Xabs.size))]
                X[np.abs(X) < threshold] = 0
            
            elif self.drop_method == "highest":

                Xabs = np.sort(np.abs(X.flat))
                threshold = Xabs[int(np.round((1-self.drop) * Xabs.size))]
                X[np.abs(X) >= threshold] = 0

            elif self.drop_method == "random":

                random.seed(self.seed)
                for idx in random.sample(range(X.size), int(self.drop * X.size)):
                    pre_idx = idx // X.shape[1]
                    post_idx = idx % X.shape[1] 
                    X[pre_idx, post_idx] = 0

        X = abs(X) if self.excitatory else X 

        t = time.time() - tstart
        info = {'time': t} # TODO: this should contain more, at least it does for LstqDrop
        return X if matrix_in or X.shape[1] > 1 else X.ravel(), info