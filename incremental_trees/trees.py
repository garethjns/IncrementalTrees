from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class StreamingRFC(RandomForestClassifier):

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
                 max_n_estimators=10):
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
        """


        super().__init__()

        self.max_n_estimators = None
        self._fit_estimators = 0
        self._estimators_per_chunk = n_estimators

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
                        max_n_estimators=max_n_estimators)

    def set_params(self,
                   **kwargs):

        # Warm start should be true to get .fit() to keep existing estimators.
        kwargs['warm_start'] = True

        for key, value in kwargs.items():
            setattr(self, key, value)

    def partial_fit(self, x, y):
        """
        Fit a single DTC using the given subset of x and y.

        Passes subset to fit, rather than using the same data each time. Wrap with Dask Incremental to handle subset
        feeding.

        :param x:
        :param y:
        :return:
        """

        # Fit the next estimator, if not done
        if self._fit_estimators < self.max_n_estimators:
            print(f"Fitting estimators {self._fit_estimators} - {self._fit_estimators + self._estimators_per_chunk} "
                  f"/ {self.max_n_estimators}")
            import time
            t0 = time.time()
            self.fit(x, y)
            t1 = time.time()
            print(f"Fit time: {round(t1 - t0, 2)}")
            print(len(self.estimators_))
            self._fit_estimators += self._estimators_per_chunk

        else:
            print('Done')
            return self


        # If still not done, prep to fit next
        if self._fit_estimators < self.max_n_estimators:
            self.n_estimators += self._estimators_per_chunk

        else:
            return self


if __name__ == '__main__':

    import pandas as pd

    data = pd.read_csv('sample_data.csv')
    x = data[[c for c in data if c != 'target']]
    y = data[['target']].values.squeeze()

    srfc = StreamingRFC(n_estimators=5,
                        max_n_estimators=100)

    for i in range(100):
        srfc.partial_fit(x, y)

    srfc.max_n_estimators