import unittest

import numpy as np
import sklearn
import sklearn.datasets

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from incremental_trees.trees import StreamingRFC
from tests.integration.base import PartialFitTests, FitTests


class TestStreamingRFC1(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(verbose=1,
                               n_estimators_per_chunk=1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFC2(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(n_estimators_per_chunk=1,
                               max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingRFC3(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFC4(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(n_estimators_per_chunk=1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFC5(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFC6(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.mod = StreamingRFC(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFC7(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 1

        cls.mod = StreamingRFC(verbose=1,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingRFC8(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 10

        cls.mod = StreamingRFC(verbose=1,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingRFC9(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 20
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 6

        cls.mod = StreamingRFC(verbose=1,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingRFR1(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(verbose=1,
                               n_estimators_per_chunk=1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFR2(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(n_estimators_per_chunk=1,
                               max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingRFR3(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFR4(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(n_estimators_per_chunk=1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFR5(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFR6(PartialFitTests, unittest.TestCase):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingRFR(n_estimators_per_chunk=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFR7(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.2
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 1

        cls.mod = StreamingRFR(verbose=1,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingRFR8(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.spf_n_fits = 10
        cls.spf_sample_prop = 0.2
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 10

        cls.mod = StreamingRFR(verbose=1,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingRFR9(FitTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""

        cls.spf_n_fits = 20
        cls.spf_sample_prop = 0.1
        cls.dask_feeding = False
        cls.n_estimators_per_sample = 6

        cls.mod = StreamingRFR(verbose=2,
                               n_estimators_per_chunk=cls.n_estimators_per_sample,
                               max_n_estimators=np.inf,
                               dask_feeding=cls.dask_feeding,
                               spf_sample_prop=cls.spf_sample_prop,
                               spf_n_fits=cls.spf_n_fits)

        super().setUpClass()


class TestStreamingEXTC1(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC2(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC3(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC4(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC5(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC6(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=int(2e4),
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


class TestStreamingEXTC7(FitTests, unittest.TestCase):
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


class TestStreamingEXTC8(FitTests, unittest.TestCase):
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


class TestStreamingEXTC9(FitTests, unittest.TestCase):
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


class TestStreamingEXTR1(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR2(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=400)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR3(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=40)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=3,
                                n_jobs=-1,
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR4(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
                                                        random_state=0,
                                                        n_features=4)

        cls.mod = StreamingEXTR(n_estimators_per_chunk=1,
                                max_features=cls.x.shape[1],
                                max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingEXTR5(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
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


class TestStreamingEXTR6(PartialFitTests, unittest.TestCase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_regression(n_samples=int(2e4),
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


class TestStreamingEXTR7(FitTests, unittest.TestCase):
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


class TestStreamingEXTR8(FitTests, unittest.TestCase):
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


class TestStreamingEXTR9(FitTests, unittest.TestCase):
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
