import time
from typing import Union

import numpy as np
import pandas as pd


class ForestAdditions:
    def partial_fit(self, X: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series],
                    classes: Union[list, np.ndarray] = None):
        """
        Fit a single DTC using the given subset of x and y.
​
        This calls .fit, which is overloaded. However flags pf_call=True, so .fit() will handle calling super .fit().
​
        For classifiers;
          - First call needs to be supplied with the expected classes (similar to existing models with .partial_fit())
            in case not all classes are present in the first subset.

        This object sets classes_ and n_classes_ depending on the supplied classes. The Individual trees set theirs
        depending on the data available in the subset. The predict_proba method is modified to standardise shape to the
        dimensions defined in this object.

        For regressors:
          - self._check_classes is overloaded with dummy method.
​
        :param x:
        :param y:
        :return:
        """
        if self.verbose > 1:
            print(f"PF Call with set classes: "
                  f"{getattr(self, 'classes_', '[no classes attr]')} and input classes {classes}")

        self._check_classes(classes=classes)

        # Fit the next estimator, if not done
        if self._fit_estimators < self.max_n_estimators:
            t0 = time.time()
            self.fit(X, y,
                     pf_call=True,
                     classes_=getattr(self, 'classes_', None))  # Pass classes for enforcement, if classifier.
            t1 = time.time()

            if self.verbose > 1:
                print(f"Fit estimators {self._fit_estimators} - {self._fit_estimators + self.n_estimators_per_chunk} "
                      f"/ {self.max_n_estimators}")
                print(f"Model reports {len(self.estimators_)}")
                print(f"Fit time: {round(t1 - t0, 2)}")
                print(len(self.estimators_))
            self._fit_estimators += self.n_estimators_per_chunk

            # If still not done, prep to fit next
            if self._fit_estimators < self.max_n_estimators:
                self.n_estimators += self.n_estimators_per_chunk

        else:
            if self.verbose > 0:
                print('Done')

        return self

    def _sampled_partial_fit(self,
                             x: Union[np.array, pd.DataFrame], y: [np.ndarray, pd.Series]):
        """
        This feeds partial_fit with random samples based on the spf_ parameters. Used by .fit() when not using dask.
        :param x: Data.
        :param y: Labels.
        :return:
        """

        n_samples = int(self.spf_sample_prop * x.shape[0])

        for _ in range(self.spf_n_fits):
            idx = np.random.randint(0, x.shape[0], n_samples)

            if self.verbose > 0:
                print(f"_sampled_partial_fit size: {idx.shape}")

            self.partial_fit(x[idx, :], y[idx],
                             classes=np.unique(y))

        return self
