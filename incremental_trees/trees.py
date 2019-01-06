from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.multiclass import _check_partial_fit_first_call
import pandas as pd
import numpy as np
from typing import Union
import time


class StreamingRFC(RandomForestClassifier):
    """
    Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params.
    """
    def __init__(self,
                 bootstrap=True,
                 class_weight=None,
                 criterion='gini',
                 max_depth=None,
                 max_features='auto',
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.0,
                 min_impurity_split=None,
                 min_samples_leaf=1,
                 min_samples_split=2,
                 min_weight_fraction_leaf=0.0,
                 n_estimators=1,
                 n_jobs=None,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 max_n_estimators=10,
                 verb: int=0) -> None:
        """

        :param bootstrap:
        :param class_weight:
        :param criterion:
        :param max_depth:
        :param max_features:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
        :param min_impurity_split:
        :param min_samples_leaf:
        :param min_samples_split:
        :param min_weight_fraction_leaf:
        :param n_estimators: Estimators per chunk to fit.
        :param n_jobs:
        :param oob_score:
        :param random_state:
        :param verbose:
        :param warm_start:
        :param max_n_estimators: Total max number of estimators to fit.
        :param verb: If > 0 display debugging info during fit
        """

        # Run the super init, which also calls other parent inits to handle other params (like base estimator)
        super().__init__()

        self.max_n_estimators: int = None
        self._fit_estimators: int = 0
        self._estimators_per_chunk: int = n_estimators
        self.classes_: np.array = None  # NB: Needs to be array, not list.
        self.n_classes_: int = None

        self.set_params(bootstrap=bootstrap,
                        class_weight=class_weight,
                        criterion=criterion,
                        max_depth=max_depth,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        min_impurity_split=min_impurity_split,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        oob_score=oob_score,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        _fit_estimators=0,
                        max_n_estimators=max_n_estimators,
                        verb=0)

    def set_params(self,
                   **kwargs) -> None:
        """
        Ensure warm_Start is set to true, otherwise set other params as usual.

        :param kwargs:
        """
        # Warm start should be true to get .fit() to keep existing estimators.
        kwargs['warm_start'] = True

        for key, value in kwargs.items():
            setattr(self, key, value)

    def partial_fit(self, x: Union[np.array, pd.DataFrame], y: Union[np.array, pd.Series],
                    classes: Union[list, np.ndarray]=None):
        """
        Fit a single DTC using the given subset of x and y.
​
        Passes subset to fit, rather than using the same data each time. Wrap with Dask Incremental to handle subset
        feeding.
​
        First call needs to be supplied with the expected classes (similar to existing models with .partial_fit())
        in case not all classes are present in the first subset.
        TODO: This currently expected every call, but alternative could be checked with the existing sklearn mechanism.
​
        Additionally, the case where not all classes are presented in the first or subsequent subsets needs to be
        handled. For the RandomForestClassifier, tree predictions are averaged in
        sklearn.ensemble.forest.accumulate_prediction unction. This sums the output matrix with dimensions
        n rows x n classes and fails if the class dimension differs.
        The class dimension is defined at the individual estimator level during the .fit() call, which sets the
        following attributes:
            - self.n_outputs_ = y.shape[1], which is then used by _validate_y_class_weight()), always called in .fit()
              to set:
                - self.classes_
                - self.n_classes_

        This object sets classes_ and n_classes_ depending on the supplied classes. The Individual trees set theirs
        depending on the data available in the subset. The predict_proba method is modified to standardise shape to the
        dimensions defined in this object.
​
        :param x:
        :param y:
        :return:
        """

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

        # Fit the next estimator, if not done
        if self._fit_estimators < self.max_n_estimators:
            t0 = time.time()
            self.fit(x, y)
            t1 = time.time()

            if self.verb > 0:
                print(f"Fit estimators {self._fit_estimators} - {self._fit_estimators + self._estimators_per_chunk} "
                      f"/ {self.max_n_estimators}")
                print(f"Fit time: {round(t1 - t0, 2)}")
                print(len(self.estimators_))
            self._fit_estimators += self._estimators_per_chunk

        else:
            if self.verb > 0:
                print('Done')
            return self

        # If still not done, prep to fit next
        if self._fit_estimators < self.max_n_estimators:
            self.n_estimators += self._estimators_per_chunk

        else:
            return self

    def predict_proba(self, x: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Call each predict proba from tree, and accumulate. This handle possibly inconsistent shapes, but isn't parallel?
​
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
        preds = np.zeros(shape=(x.shape[0], self.n_classes_ + 1),
                         dtype=np.float32)

        for e in self.estimators_:
            # Get the prediction from the tree
            est_preds = e.predict_proba(x)
            # Get the indexes of the classes present
            present_classes = e.classes_.astype(int)
            # Sum these in to the correct array columns
            preds[:, present_classes] += est_preds

        return preds / len(self.estimators_)


if __name__ == '__main__':

    data = pd.read_csv('sample_data.csv')
    x = data[[c for c in data if c != 'target']]
    y = data[['target']].values.squeeze()

    srfc = StreamingRFC(n_estimators=5,
                        max_n_estimators=100)

    for i in range(100):
        srfc.partial_fit(x, y)

    srfc.max_n_estimators
