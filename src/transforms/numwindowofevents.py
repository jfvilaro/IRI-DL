from __future__ import division

import numpy as np
import heapq
from src.transforms.transforms import TransformBase


class WindowNumberSAE(TransformBase):
    """Calculate Surface of N active events
    """

    def __init__(self, perkey_args, general_args):
        super(WindowNumberSAE, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]

                if isinstance(elem, list):
                    for i, el in enumerate(elem):
                        elem[i] = self._compute_exponential_decay_ts(elem)
                else:
                    sample[sample_key] = self._compute_window(elem, self._params['img1']['window_size'])

        return sample

    def _compute_window(self, patch, window_size):

        num_of_pixels = patch.shape[0]*patch.shape[1]
        patch_size = patch.shape[0]

        flat_patch = np.reshape(patch, num_of_pixels)

        number_of_nonzero_elements = np.argwhere(flat_patch > 0).shape[0] # Check number of nonzero elements
        new_window_size = min(window_size, number_of_nonzero_elements) # Ensure that we are not using a window that uses 0 elements
        inlier_events_id = heapq.nlargest(new_window_size, range(flat_patch.size),patch.take)  # Select the newest N-events

        bottom_val = np.amin(flat_patch[inlier_events_id])  # Find minimum value of the chosen events
        flat_out_patch = np.ones(num_of_pixels)*bottom_val  # Define new patch
        flat_out_patch[inlier_events_id] = flat_patch[inlier_events_id]  # Add inlier values to the new patch

        flat_norm_sae = (flat_out_patch-np.amin(flat_out_patch))/(np.amax(flat_out_patch)-np.amin(flat_out_patch)) # normalize between 0 and 1

        preprocessed_sae = np.reshape(flat_norm_sae, (patch_size, patch_size))  # Reshape output

        return preprocessed_sae


    def __str__(self):
        return 'Calculate Speed Invariant Time surface:' + str(self._params)
