import warnings
from typing import Union

import numpy as np
import pandas as pd

from incremental_trees.add_ins.forest_overloads import ForestOverloads


class ClassifierOverloads(ForestOverloads):
    """
    Overloaded methods specific to classifiers.
    """

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Call each predict proba from tree, and accumulate. This handle possibly inconsistent shapes, but isn't
        parallel?
    ​
        Cases where not all classes are presented in the first or subsequent subsets needs to be
        handled. For the RandomForestClassifier, tree predictions are averaged in
        sklearn.ensemble.forest.accumulate_prediction function. This sums the output matrix with dimensions
        n rows x n classes and fails if the class dimension differs.
        The class dimension is defined at the individual estimator level during the .fit() call, which sets the
        following attributes:
            - self.n_outputs_ = y.shape[1], which is then used by _validate_y_class_weight()), always called in .fit()
              to set:
                - self.classes_
                - self.n_classes_

        The .predict() method (sklearn.tree.tree.BaseDecisionTree.predict()) sets the output shape using:
            # Classification
            if is_classifier(self):
                if self.n_outputs_ == 1:
                    return self.classes_.take(np.argmax(proba, axis=1), axis=0)
                else:
                   [Not considering this yet]

        :param x:
        :return:
        """
        # Prepare expected output shape
        preds = np.zeros(shape=(x.shape[0], self.n_classes_),
                         dtype=np.float32)
        counts = np.zeros(shape=(x.shape[0], self.n_classes_),
                          dtype=np.int16)

        for e in self.estimators_:
            # Get the prediction from the tree
            est_preds = e.predict_proba(x)
            # Get the indexes of the classes present
            present_classes = e.classes_.astype(int)
            # Sum these in to the correct array columns
            preds[:, present_classes] += est_preds
            counts[:, present_classes] += 1

        # Normalise predictions against counts
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            norm_prob = preds / counts

        # And remove nans (0/0) and infs (n/0)
        norm_prob[np.isnan(norm_prob) | np.isinf(norm_prob)] = 0

        return norm_prob
