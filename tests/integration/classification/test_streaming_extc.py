import numpy as np
from dask_ml.wrappers import Incremental
from sklearn.datasets import make_blobs

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from tests.integration.base.class_consistency_test_base import ClassConsistencyTestBase
from tests.integration.base.dask_test_base import DaskTestBase
from tests.integration.base.fit_test_base import FitTestBase
from tests.integration.base.partial_fit_test_base import PartialFitTestBase


class TestStreamingEXTCWithPartialFitsUnlimitedEstimators(PartialFitTestBase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    No limit on the total number of trees.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = make_blobs(n_samples=int(2e4), random_state=0, n_features=40, centers=2, cluster_std=100)
        cls.mod = StreamingEXTC(n_estimators_per_chunk=1, max_n_estimators=np.inf)
        cls.expected_n_estimators = 100

        super().setUpClass()


class TestStreamingEXTCWithPartialFitsLimitedEstimators(PartialFitTestBase):
    """
    Test SEXT with single estimator per chunk with "random forest style" max features. ie, subset.

    Total models limited to 39.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = make_blobs(n_samples=int(2e4), random_state=0, n_features=40, centers=2, cluster_std=100)
        cls.mod = StreamingEXTC(n_estimators_per_chunk=1, max_n_estimators=39)
        cls.expected_n_estimators = 39

        super().setUpClass()


class TestStreamingEXTCWithPartialFitsMultipleEstimatorsPerChunk(PartialFitTestBase):
    """
    Test SEXT with multiple estimators per chunk with "random forest style" max features. ie, subset.

    No limit on total models, 3 estimators per row subset (each with different feature subset)
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = make_blobs(n_samples=int(2e4), random_state=0, n_features=40, centers=2, cluster_std=100)
        cls.mod = StreamingEXTC(n_estimators_per_chunk=3, n_jobs=-1, max_n_estimators=np.inf)
        cls.expected_n_estimators = 300

        super().setUpClass()


class TestStreamingEXTCWithPartialFitsAllFeatures(PartialFitTestBase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 1 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.n_samples = 1000
        cls.x, cls.y = make_blobs(n_samples=int(2e4), random_state=0, n_features=40, centers=2, cluster_std=100)
        cls.mod = StreamingEXTC(n_estimators_per_chunk=1, max_features=cls.x.shape[1], max_n_estimators=np.inf)
        cls.expected_n_estimators = 100

        super().setUpClass()


class TestStreamingEXTCWithPartialFitsMultipleEstimatorsPerChunkAllFeatures(PartialFitTestBase):
    """
    Test SEXT with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

    No limit on total models, 3 estimators per row subset.
    """

    @classmethod
    def setUpClass(cls):
        cls.n_samples = 1000
        cls.x, cls.y = make_blobs(n_samples=int(2e4), random_state=0, n_features=40, centers=2, cluster_std=100)

        cls.mod = StreamingEXTC(
            n_estimators_per_chunk=3,
            n_jobs=-1,
            max_features=cls.x.shape[1],
            max_n_estimators=np.inf
        )
        cls.expected_n_estimators = 300

        super().setUpClass()


class TestStreamingEXTCWithFitSingleEstimatorPerChunk(FitTestBase):
    @classmethod
    def setUpClass(cls):
        cls.spf_n_fits = 10
        cls.n_estimators_per_sample = 1
        cls.mod = StreamingEXTC(
            verbose=1,
            n_estimators_per_chunk=cls.n_estimators_per_sample,
            max_n_estimators=np.inf,
            dask_feeding=cls.dask_feeding,
            spf_sample_prop=cls.spf_sample_prop,
            spf_n_fits=cls.spf_n_fits
        )

        super().setUpClass()


class TestStreamingEXTCWithFitMultipleEstimatorsPerChunk(FitTestBase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls.spf_n_fits = 10
        cls.n_estimators_per_sample = 10
        cls.mod = StreamingEXTC(
            verbose=1,
            n_estimators_per_chunk=cls.n_estimators_per_sample,
            max_n_estimators=np.inf,
            dask_feeding=cls.dask_feeding,
            spf_sample_prop=cls.spf_sample_prop,
            spf_n_fits=cls.spf_n_fits
        )

        super().setUpClass()


class TestStreamingEXTCWithFitAdditionalSteps(FitTestBase):
    @classmethod
    def setUpClass(cls):
        cls.spf_n_fits = 20
        cls.n_estimators_per_sample = 6
        cls.mod = StreamingEXTC(
            verbose=1,
            n_estimators_per_chunk=cls.n_estimators_per_sample,
            max_n_estimators=np.inf,
            dask_feeding=cls.dask_feeding,
            spf_sample_prop=cls.spf_sample_prop,
            spf_n_fits=cls.spf_n_fits
        )

        super().setUpClass()


class TestDaskEXTCWithDask(DaskTestBase):
    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls, reg=False)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=1, max_n_estimators=39, verbose=1))
        cls.expected_n_estimators = 10

        super().setUpClass()


class TestDaskEXTCWithDaskMultipleEstimatorsPerChunk(DaskTestBase):
    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls, reg=False)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=2, n_jobs=-1, max_n_estimators=np.inf, verbose=1))
        cls.expected_n_estimators = 20

        super().setUpClass()


class TestDaskEXTCWithDaskManyEstimatorsPerChunk(DaskTestBase):
    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls, reg=False)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=20, n_jobs=-1, max_n_estimators=np.inf, verbose=1))
        cls.expected_n_estimators = 200

        super().setUpClass()


class TestDaskEXTCWithDaskAlLFeatures(DaskTestBase):
    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls, reg=False)
        cls.mod = Incremental(
            StreamingEXTC(
                n_estimators_per_chunk=1, n_jobs=-1,
                max_n_estimators=np.inf,
                max_features=cls.x.shape[1],
                verbose=1
            )
        )
        cls.expected_n_estimators = 10

        super().setUpClass()


class TestInconsistentClassesEXTC(ClassConsistencyTestBase):
    def setUp(self):
        self.mod = StreamingEXTC(n_estimators_per_chunk=1, max_n_estimators=np.inf, verbose=2)


del FitTestBase, PartialFitTestBase, DaskTestBase, ClassConsistencyTestBase
