import unittest
import numpy as np
from incremental_trees.trees import StreamingEXTC, StreamingEXTR
import sklearn
import sklearn.datasets
from tests.test_streaming_forest import PartialFitTests, FitTests


class TestStreamingEXTC_1(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_2(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=1,
                                max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_3(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_4(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_5(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_6(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e5),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingEXTC(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTC_7(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 1

        cls.mod = StreamingEXTC(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTC_8(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 10

        cls.mod = StreamingEXTC(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTC_9(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 20
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 6

        cls.mod = StreamingEXTC(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTR_1(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_2(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=400)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_3(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_4(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=4)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_5(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_6(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e5),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR_7(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 1

        cls.mod = StreamingEXTR(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTR_8(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 10

        cls.mod = StreamingEXTR(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTR_9(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 20
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 6

        cls.mod = StreamingEXTR(verbose=1,
                                n_estimators_per_chunk=cls.n_estimators_per_sample,
                                max_n_estimators=np.inf,
                                dask_feeding=cls.dask_feeding,
                                spf_sample_prop=cls.spf_sample_prop,
                                spf_n_fits=cls.spf_n_fits)

        super().setUpClass()