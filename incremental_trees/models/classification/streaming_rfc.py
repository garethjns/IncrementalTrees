from typing import Optional, Union, Dict, List

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from incremental_trees.add_ins.classifier_additions import ClassifierAdditions
from incremental_trees.add_ins.classifier_overloads import ClassifierOverloads


class StreamingRFC(ClassifierAdditions, ClassifierOverloads, RandomForestClassifier):
    """
    Overload sklearn.ensemble.RandomForestClassifier to add partial fit method and new params.

    Note this init is a slightly different structure to ExtraTressClassifier/Regressor and RandomForestRegressor.
    """

    def __init__(self,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_samples_leaf: int = 1,
                 min_weight_fraction_leaf: float = 0.0,
                 max_features: Optional[str] = 'sqrt',
                 max_leaf_nodes: Optional[int] = None,
                 min_impurity_decrease: float = 0.0,
                 bootstrap: bool = True,
                 oob_score: bool = False,
                 n_jobs: Optional[int] = None,
                 random_state: Optional[int] = None,
                 verbose: int = 0,
                 warm_start: bool = True,
                 class_weight: Optional[Union[str, Dict, List[Dict]]] = None,
                 ccp_alpha: float = 0.0,
                 max_samples: Optional[int] = None,
                 dask_feeding: bool = True,
                 n_estimators_per_chunk: int = 1,
                 max_n_estimators: int = 10,
                 spf_n_fits: int = 100,
                 spf_sample_prop: float = 0.1) -> None:
        """
        :param bootstrap:
        :param class_weight:
        :param criterion:
        :param max_depth:
        :param max_features:
        :param max_leaf_nodes:
        :param min_impurity_decrease:
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

        self.set_params(bootstrap=bootstrap,
                        class_weight=class_weight,
                        criterion=criterion,
                        max_depth=max_depth,
                        max_features=max_features,
                        max_leaf_nodes=max_leaf_nodes,
                        min_impurity_decrease=min_impurity_decrease,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split,
                        min_weight_fraction_leaf=min_weight_fraction_leaf,
                        n_estimators_per_chunk=n_estimators_per_chunk,
                        n_estimators=n_estimators_per_chunk,
                        n_jobs=n_jobs,
                        oob_score=oob_score,
                        random_state=random_state,
                        verbose=verbose,
                        warm_start=warm_start,
                        _fit_estimators=0,
                        dask_feeding=dask_feeding,
                        max_n_estimators=max_n_estimators,
                        verb=0,
                        spf_n_fits=spf_n_fits,
                        spf_sample_prop=spf_sample_prop,
                        ccp_alpha=ccp_alpha,
                        max_samples=max_samples
                        )
