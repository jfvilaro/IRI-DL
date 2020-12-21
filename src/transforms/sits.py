from __future__ import division

import numpy as np
from src.transforms.transforms import TransformBase


class SITS(TransformBase):
    """Calculate Speed Invariant Time Surface
    """

    def __init__(self, perkey_args, general_args):
        super(SITS, self).__init__()
        self._params = self._find_params_for_sample_key(perkey_args, general_args)

    def __call__(self, sample):
        for sample_key in sample.keys():
            if sample_key in self._params.keys():
                elem = sample[sample_key]

                if isinstance(elem, list):
                    for i, el in enumerate(elem):
                        elem[i] = self._compute_speed_invariant_ts(elem)
                else:
                    sample[sample_key] = self._compute_speed_invariant_ts(elem)

        return sample

    def _compute_speed_invariant_ts(self,patch):

        num_of_pixels = patch.shape[0] * patch.shape[1]
        patch_size = patch.shape[0]

        flat_patch = np.reshape(patch, num_of_pixels)  # Reshape the patch as one dimensional array
        central_idc = (flat_patch.shape[0] - 1) / 2  # central pixel position
        positions_with_events_id = np.where(flat_patch != 0)[0]  # Find non-empty positions
        order_of_events = positions_with_events_id[np.argsort(-flat_patch[positions_with_events_id])]  # Sort in decreasing order
        new_order_of_events = np.concatenate((np.array([central_idc]), np.delete(order_of_events, np.where(order_of_events == central_idc)))).astype(int)  # ensure that the central event is the first in the decreasing order (for the cases where there are events with same ts)

        sits_flat_patch = np.zeros(num_of_pixels)  # Define speed invariant time surface array

        # Sort by time (decreasing order) all the events and give as value the position in the sorted list

        for i in range(new_order_of_events.size):
            sits_flat_patch[new_order_of_events[i]] = num_of_pixels - i

        preprocessed_sae = np.reshape(sits_flat_patch, (patch_size, patch_size))

        # Check max initial position and final position, should be always center pixel
        #max_positions_init = np.argwhere(patch == np.amax(patch))
        #max_positions_init_flat = np.argwhere(flat_patch == np.amax(flat_patch))
        #max_positions_end = np.argwhere(preprocessed_sae == np.amax(preprocessed_sae))
        #max_positions_end_flat = np.argwhere(sits_flat_patch == np.amax(sits_flat_patch))

        return preprocessed_sae

    def __str__(self):
        return 'Calculate Speed Invariant Time surface:' + str(self._params)
