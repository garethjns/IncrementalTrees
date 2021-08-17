from typing import Optional

import numpy as np


class ForestOverloads:
    def set_params(self,
                   **kwargs):
        """
        Ensure warm_Start is set to true, otherwise set other params as usual.

        :param kwargs: Params to set.
        """
        # Warm start should be true to get .fit() to keep existing estimators.
        kwargs['warm_start'] = True

        for key, value in kwargs.items():
            setattr(self, key, value)

        return self

    def fit(self, *args, pf_call: bool = False, classes_: Optional[np.ndarray] = None,
            sample_weight: Optional[np.array] = None, **kwargs):
        """
        This fit handles calling either super().fit or partial_fit depending on the caller.

        :param pf_call: True if called from partial fit, in this case super.fit() is called, instead of getting stuck in
                        a recursive loop.
        :param classes_: On pf calls, classes is passed from self.classes which will have already been set. These are
                         re-set after the call to super's fit, which will change them based on observed data.
        :param sample_weight: Sample weights. If None, then samples are equally weighted.
        """

        if not self.dask_feeding and not pf_call:
            if self.verbose > 0:
                print('Feeding with spf')
            self._sampled_partial_fit(*args, sample_weight=sample_weight, **kwargs)

        else:

            if self.verbose > 0:
                print('Fitting from a partial_fit call')
            super().fit(*args, sample_weight=sample_weight, **kwargs)
            if classes_ is not None:
                self.classes_ = classes_
                self.n_classes_ = len(classes_)

        return self
