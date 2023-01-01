from typing import Union

import sklearn
from sklearn import clone
from sklearn.model_selection import RandomizedSearchCV

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from tests.integration.base.predict_test_base import PredictTestBase


class FitTestBase(PredictTestBase):
    """
    Test direct calls to.fit with dask off, which will use ._sampled_partial_fit() to feed partial_fit.
    """

    spf_n_fits: int
    n_samples: int
    n_estimators_per_sample: int
    mod: Union[StreamingEXTC, StreamingEXTR, StreamingRFC, StreamingRFR]
    dask_feeding: bool = False
    spf_sample_prop: float = 0.1

    @classmethod
    def setUpClass(cls):
        """
        Set helper actual model from specified values. Need to super this from child setUpClass.
        :return:
        """
        cls.expected_n_estimators = cls.spf_n_fits * cls.n_estimators_per_sample

        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=cls.n_samples,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.grid = RandomizedSearchCV(clone(cls.mod),
                                      scoring='roc_auc',
                                      cv=2,
                                      n_iter=3,
                                      verbose=10,
                                      param_distributions={'spf_sample_prop': [0.1, 0.2, 0.3],
                                                           'spf_n_fits': [10, 20, 30]},
                                      n_jobs=-1)

    def test_fit__sampled_partial_fit(self):
        """With dask off, call .fit directly."""
        self.mod.fit(self.x, self.y)

    def test_n_estimators(self):
        self.assertEqual(self.expected_n_estimators, len(self.mod.estimators_))

    def test_grid_search(self):
        """With dask off, try with sklearn GS."""
        self.grid.fit(self.x, self.y)
        self.grid.score(self.x, self.y)
