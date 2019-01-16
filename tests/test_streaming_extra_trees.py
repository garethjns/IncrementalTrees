import unittest
import numpy as np
from incremental_trees.trees import StreamingEXT
import dask_ml
import dask_ml.datasets
from tests.test_streaming_forest import Common


class TestStreamingEXT_1(Common, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXT_2(Common, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=1,
                               max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingEXT_3(Common, unittest.TestCase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXT_4(Common, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXT_5(Common, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXT_5(Common, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXT(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()
