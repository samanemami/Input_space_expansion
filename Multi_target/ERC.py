# Author: Seyedsaman Emami
# Licence: GNU Lesser General Public License v2.1 (LGPL-2.1)

import pickle
import numpy as np
import multiprocessing as mpc
from Base import _base


class erc(_base.BaseEstimator):
    def __init__(self,
                 model,
                 verbose):

        self.model = model
        self.verbose = verbose

    def _training(self):
        pass
