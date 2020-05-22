from typing import List

import numpy as np

from incremental_trees.add_ins.forest_additions import ForestAdditions
from incremental_trees.add_ins.sklearn_overloads import _check_partial_fit_first_call


class ClassifierAdditions(ForestAdditions):
    """
    Additional functions specific to classifiers.
    """

    def _check_classes(self, classes: List[int]):
        """Set classes if they haven't been set yet, otherwise do nothing."""

        # Set classes for forest (this only needs to be done once).
        # Not for each individual tree, these will be set by .fit() using the classes available in the subset.
        # Check classes_ is set, or provided
        # Returns false if nothing to do
        classes_need_setting = _check_partial_fit_first_call(self, classes)

        # If classes not set, set
        # Above will error if not set and classes = None
        if classes_need_setting:
            self.classes_ = np.array(classes)
            self.n_classes_ = len(classes)
