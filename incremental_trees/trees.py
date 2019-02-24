from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import ExtraTreeClassifier, ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.utils.multiclass import unique_labels
import pandas as pd
import numpy as np
from typing import Union, List
import time
import warnings


def _check_partial_fit_first_call(clf,
                                  classes=None):
    """
    Modified sklearn function. If classes are inconsistent on second call, warn and reuse previous
    on assumption first call specification was correct. Don't raise error.

    Private helper function for factorizing common classes param logic

    Estimators that implement the ``partial_fit`` API need to be provided with
    the list of possible classes at the first call to partial_fit.

    Modification:
    Subsequent calls to partial_fit do not check () that ``classes`` is still
    consistent with a previous value of ``clf.classes_`` when provided.

    This function returns True if it detects that this was the first call to
    ``partial_fit`` on ``clf``. In that case the ``classes_`` attribute is also
    set on ``clf``.
    """

    if getattr(clf, 'classes_', None) is None and classes is None:
        raise ValueError("classes must be passed on the first call "
                         "to partial_fit.")

    elif classes is not None:
        if getattr(clf, 'classes_', None) is not None:
            if not np.array_equal(clf.classes_, unique_labels(classes)):
                # Don't error here:
                # Instead, use the previous classes setting, which must be correct on first setting
                warnings.warn(f"Classes differ on this call, ignoring on the assumption first call was correct.")
                return True
                # raise ValueError(
                #     "`classes=%r` is not the same as on last call "
                #     "to partial_fit, was: %r" % (classes, clf.classes_))

        else:
            # This is the first call to partial_fit
            clf.classes_ = unique_labels(classes)
            return True

    # classes is None and clf.classes_ has already previously been set:
    # nothing to do
    return False


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

        self._check_classes(classes=classes)

        # Fit the next estimator, if not done
        if self._fit_estimators < self.max_n_estimators:
            t0 = time.time()
            self.fit(X, y,
                     pf_call=True)
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

        for _ in range(self.spf_n_fits):
            idx = np.random.randint(0, x.shape[0], self.spf_n_samples)
            self.partial_fit(x[idx, :], y[idx],
                             classes=np.unique(y))

        return self


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


class RegressorAdditions(ForestAdditions):
    def _check_classes(self, **kwargs) -> None:
        """
        Don't need to check classes with the regressor.
        """
        pass


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

    def fit(self, *args,
            pf_call=False):
        """
        This fit handles calling either super().fit or partial_fit depending on the caller.

        :param pf_call: True if called from partial fit, in this case super.fit() is called, instead of getting stuck in
                        a recursive loop.
        """

        if not self.dask_feeding and not pf_call:
            if self.verbose > 0:
                print('Feeding with spf')
            self._sampled_partial_fit(*args)

        else:

            if self.verbose > 0:
                print('Fitting from a partial_fit call')
            super().fit(*args)

        return self


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
        preds = np.zeros(shape=(x.shape[0], self.n_classes_ + 1),
                         dtype=np.float32)
        counts = np.zeros(shape=(x.shape[0], self.n_classes_ + 1),
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


class RegressorOverloads(ForestOverloads):
    """
    Nothing specific to overload for the Regressors. Predict doesn't need to deal with classes.
    """
    pass


class StreamingRFR(RegressorAdditions, RegressorOverloads, RandomForestRegressor):
    """Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params."""
    def __init__(self,
                 n_estimators='warn',
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 n_estimators_per_chunk: int=1,
                 warm_start: bool=True,
                 dask_feeding: bool=True,
                 max_n_estimators=10,
                 spf_on=False,
                 spf_n_fits=100,
                 spf_n_samples=100):

        super(RandomForestRegressor, self).__init__(
            base_estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators_per_chunk,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self._fit_estimators = 0
        self.max_n_estimators = max_n_estimators
        self.n_estimators_per_chunk = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # Set additional params.
        self.set_params(n_estimators_per_chunk=n_estimators_per_chunk,
                        spf_on=spf_on,
                        spf_n_fits=spf_n_fits,
                        spf_n_samples=spf_n_samples,
                        dask_feeding=dask_feeding)


class StreamingRFC(ClassifierAdditions, ClassifierOverloads, RandomForestClassifier):
    """
    Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params.

    Note this init is a slightly different structure to ExtraTressClassifier/Regressor and RandomForestRegressor.
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
                 n_estimators_per_chunk: int=1,
                 n_estimators: bool=None,
                 n_jobs=None,
                 oob_score=False,
                 random_state=None,
                 verbose=0,
                 warm_start: bool=True,
                 dask_feeding: bool=True,
                 max_n_estimators=10,
                 spf_on=False,
                 spf_n_fits=100,
                 spf_n_samples=100) -> None:
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
        :param n_estimators_per_chunk: Estimators per chunk to fit.
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
        self.classes_: np.array = None  # NB: Needs to be array, not list.
        self.n_classes_: int = None

        # n_estimators will be used by RFC to set how many ests are fit on each .fit() call
        n_estimators = n_estimators_per_chunk

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
                        n_estimators_per_chunk=n_estimators_per_chunk,
                        n_estimators=n_estimators,
                        n_jobs=n_jobs,
                        oob_score=oob_score,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        _fit_estimators=0,
                        dask_feeding=dask_feeding,
                        max_n_estimators=max_n_estimators,
                        verb=0,
                        spf_on=spf_on,
                        spf_n_fits=spf_n_fits,
                        spf_n_samples=spf_n_samples)


class StreamingEXTR(RegressorAdditions, RegressorOverloads, ExtraTreesRegressor):
    def __init__(self,
                 n_estimators_per_chunk: int=1,
                 n_estimators='warn',
                 max_n_estimators=np.inf,
                 criterion="mse",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=True,
                 dask_feeding: bool=True,
                 spf_on: bool=False,
                 spf_n_fits: int=100,
                 spf_n_samples: int=100):

        super(ExtraTreesRegressor, self).__init__(
            base_estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators_per_chunk,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self._fit_estimators = 0
        self.max_n_estimators = max_n_estimators
        self.n_estimators_per_chunk = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # Set additional params.
        self.set_params(n_estimators_per_chunk=n_estimators_per_chunk,
                        max_n_estimators=max_n_estimators,
                        spf_on=spf_on,
                        spf_n_fits=spf_n_fits,
                        spf_n_samples=spf_n_samples,
                        dask_feeding=dask_feeding)


class StreamingEXTC(ClassifierAdditions, ClassifierOverloads, ExtraTreesClassifier):
    """Overload sklearn.ensemble.ExtraTreesClassifier to add partial fit method and new params."""
    def __init__(self,
                 n_estimators_per_chunk: int=1,
                 n_estimators: bool=None,
                 max_n_estimators=np.inf,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=False,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 warm_start=True,
                 class_weight=None,
                 dask_feeding: bool=True,
                 spf_on=False,
                 spf_n_fits=100,
                 spf_n_samples=100):

        super(ExtraTreesClassifier, self).__init__(
            base_estimator=ExtraTreeClassifier(),
            n_estimators=n_estimators_per_chunk,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self._fit_estimators = 0
        self.max_n_estimators = max_n_estimators
        self.n_estimators_per_chunk = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # Set additional params.
        self.set_params(n_estimators_per_chunk=n_estimators_per_chunk,
                        max_n_estimators=max_n_estimators,
                        spf_on=spf_on,
                        spf_n_fits=spf_n_fits,
                        spf_n_samples=spf_n_samples,
                        dask_feeding=dask_feeding)


if __name__ == '__main__':
    from sklearn.datasets import make_blobs, make_regression

    x, y = make_regression(n_samples=int(2e5),
                           random_state=0,
                           n_features=40)

    srfr = StreamingRFR(n_estimators_per_chunk=5,
                        spf_n_fits=10,
                        spf_on=True,
                        verbose=0,
                        n_jobs=2)

    # Fit 10 regressors
    for _ in range(10):
        x, y = make_regression(n_samples=int(2e5),
                               random_state=0,
                               n_features=40)

        srfr = StreamingRFR(n_estimators_per_chunk=5,
                            max_n_estimators=100,
                            verbose=0,
                            n_jobs=5)

        chunk_size = int(2e3)
        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            srfr.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SRFR: {srfr.score(x, y)}")

        sext = StreamingEXTR(n_estimators_per_chunk=5,
                             max_n_estimators=100,
                             verbose=0,
                             n_jobs=5)

        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            sext.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SEXTR: {sext.score(x, y)}")

    # Fit 10 classifiers
    for _ in range(10):
        x, y = make_blobs(n_samples=int(2e5),
                          random_state=0,
                          n_features=40,
                          centers=2,
                          cluster_std=100)

        srfc = StreamingRFC(n_estimators_per_chunk=5,
                            max_n_estimators=100,
                            verbose=0,
                            n_jobs=5)

        chunk_size = int(2e3)
        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            srfc.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SRFC: {srfc.score(x, y)}")

        sext = StreamingEXTC(n_estimators_per_chunk=5,
                             max_n_estimators=100,
                             verbose=0,
                             n_jobs=5)

        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            sext.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SEXTC: {sext.score(x, y)}")
