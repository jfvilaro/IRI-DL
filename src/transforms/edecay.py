from __future__ import division

import numpy as np
import math
from src.transforms.transforms import TransformBase


class ExponentialDecaySAE(TransformBase):
    """Calculate Exponential time decay SAE
    """

    def __init__(self, perkey_args, general_args):
        super(ExponentialDecaySAE, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]

                if isinstance(elem, list):
                    for i, el in enumerate(elem):
                        elem[i] = self._compute_exponential_decay_ts(elem)
                else:
                    sample[sample_key] = self._compute_exponential_decay_ts(elem)

        return sample

    def _compute_exponential_decay_ts(self, patch):

        current_time = np.amax(patch)  # time stamp of the current event
        # For each pixel of the patch compute exponential decay
        exp_decay_v = np.vectorize(self._exponential_decay)
        preprocessed_sae = exp_decay_v(patch, self._params['img1']['tau'],current_time)

        # Check max initial position and final position, should be always center pixel
        # max_positions_init = np.argwhere(patch == np.amax(patch))
        # max_positions_end = np.argwhere(preprocessed_sae == np.amax(preprocessed_sae))

        return preprocessed_sae

    def _exponential_decay(self, n, tau, current_time):
        return math.exp(-(current_time - n) / tau)

    def __str__(self):
        return 'Calculate Speed Invariant Time surface:' + str(self._params)
