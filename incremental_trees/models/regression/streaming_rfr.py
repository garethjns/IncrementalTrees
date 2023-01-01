from typing import Optional

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from incremental_trees.add_ins.regressor_additions import RegressorAdditions
from incremental_trees.add_ins.regressor_overloads import RegressorOverloads


class StreamingRFR(RegressorAdditions, RegressorOverloads, RandomForestRegressor):
    """Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params."""

    def __init__(self,
                 criterion: str = "squared_error",
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: float = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Optional[float] = 1.0,
                 max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 verbose: int = 0,
                 n_estimators_per_chunk: int = 1,
                 warm_start: bool = True,
                 dask_feeding: bool = True,
                 max_n_estimators: int = 10,
                 spf_n_fits: int = 100,
                 spf_sample_prop: float = 0.1):
        super(RandomForestRegressor, self).__init__(
            estimator=DecisionTreeRegressor(),
            n_estimators=n_estimators_per_chunk,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self._fit_estimators = 0
        self.max_n_estimators = max_n_estimators
        self.n_estimators_per_chunk = n_estimators_per_chunk
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease

        # Set additional params.
        self.set_params(n_estimators_per_chunk=n_estimators_per_chunk,
                        spf_n_fits=spf_n_fits,
                        spf_sample_prop=spf_sample_prop,
                        dask_feeding=dask_feeding)
