from typing import Optional, Union

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import ExtraTreeRegressor

from incremental_trees.add_ins.regressor_additions import RegressorAdditions
from incremental_trees.add_ins.regressor_overloads import RegressorOverloads


class StreamingEXTR(RegressorAdditions, RegressorOverloads, ExtraTreesRegressor):
    def __init__(self,
                 criterion: str = "squared_error",
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Union[str, float] = 1.0,
                 max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = False,
                 oob_score: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 verbose: int = 0,
                 warm_start: bool = True,
                 ccp_alpha: float = 0.0,
                 max_samples: Optional[float] = None,
                 max_n_estimators: int = np.inf,
                 n_estimators_per_chunk: int = 1,
                 dask_feeding: bool = True,
                 spf_n_fits: int = 100,
                 spf_sample_prop: float = 0.1):
        super(ExtraTreesRegressor, self).__init__(
            estimator=ExtraTreeRegressor(),
            n_estimators=n_estimators_per_chunk,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "random_state", "ccp_alpha"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples
        )

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
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

        # Set additional params.
        self.set_params(n_estimators_per_chunk=n_estimators_per_chunk,
                        max_n_estimators=max_n_estimators,
                        spf_n_fits=spf_n_fits,
                        spf_sample_prop=spf_sample_prop,
                        dask_feeding=dask_feeding)
