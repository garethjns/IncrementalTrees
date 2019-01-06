import unittest
import numpy as np
from incremental_trees.trees import StreamingRFC
from sklearn.datasets import make_blobs
from dask.distributed import Client, LocalCluster
import dask_ml
import dask_ml.datasets
from dask_ml.wrappers import Incremental


class Common(unittest.TestCase):
    """
    Standard tests to run on supplied model and data.

    Inherit this into a class with model/data defined in setUpClass into self.mod, self.x, self.y. Then call the
    setupClass method here to set some helper values.

    These tests need to run in order, as self.mod used through tests. Maybe would be better to mock it each time,
    but lazy....

    These are run without using Dask, so the subset passing to partial_fit is handled manually.
    """
    @classmethod
    def setUpClass(cls):
        """
        Set helper values from specified model/data. Need to super this from child setUpClass.
        :return:
        """
        cls.chunk_size = 10
        cls.n_chunks = int(cls.n_samples / cls.chunk_size)
        cls.samples_per_chunk = int(cls.n_samples / cls.n_chunks)

        # Cursor will be tracked through data between tests.
        cls.s_idx = 0
        cls.e_idx = cls.samples_per_chunk

    def test_first_partial_fit_call(self):
        """
        Call partial_fit for the first time on self.mod.
        :return:
        """
        # Call the first partial fit specifying classes
        self.mod.partial_fit(self.x[self.s_idx:self.e_idx, :],
                             self.y[self.s_idx:self.e_idx],
                             classes=np.unique(self.y))

    def test_next_partial_fit_calls(self):
        """
        Call partial fit on remaining chunks.

        Provide classes again on second iteration, otherwise don't.

        :return:
        """
        for i in range(1, self.n_chunks):
            self.mod.partial_fit(self.x[self.s_idx:self.e_idx, :],
                                 self.y[self.s_idx:self.e_idx],
                                 classes=np.unique(self.y) if i == 2 else None)

            self.s_idx = self.e_idx
            self.e_idx = self.s_idx + self.samples_per_chunk

        # Set expected number of esitmators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod._estimators_per_chunk * self.n_chunks), self.mod.max_n_estimators)
        self.assertEqual(expect_, self.expected_n_estimators)
        # Then check the model matches the validated expectation
        self.assertEqual(len(self.mod.estimators_), self.expected_n_estimators)

    def test_predict(self):
        """
        Test prediction function runs are returns expected shape, even if all classes are not in prediction set.
        :return:
        """

        # Predict on all data
        self.mod.predict(self.x)

        # Predict on single row
        self.mod.predict(self.x[0, :].reshape(1, -1))

    def test_result(self):
        """Test performance of model is approximately as expected."""
        pass


class TestStreamingRFC_1(Common):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

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

        cls.mod = StreamingRFC(n_estimators=1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFC_2(Common):
    """
    Test SRFC with single estimator per chunk with "random forest style" max features. ie, subset.

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

        cls.mod = StreamingRFC(n_estimators=1,
                               max_n_estimators=39)

        # Set expected number of estimators
        cls.expected_n_estimators = 39

        # Set helper values
        super().setUpClass()


class TestStreamingRFC_3(Common):
    """
    Test SRFC with multiple estimators per chunk with "random forest style" max features. ie, subset.

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

        cls.mod = StreamingRFC(n_estimators=3,
                               n_jobs=-1,
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()


class TestStreamingRFC_4(Common):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

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

        cls.mod = StreamingRFC(n_estimators=1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 100

        # Set helper values
        super().setUpClass()


class TestStreamingRFC_5(Common):
    """
    Test SRFC with single estimator per chunk with "decision tree style" max features. ie, all available to each tree.

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

        cls.mod = StreamingRFC(n_estimators=3,
                               n_jobs=-1,
                               max_features=cls.x.shape[1],
                               max_n_estimators=np.inf)

        # Set expected number of estimators
        cls.expected_n_estimators = 300

        # Set helper values
        super().setUpClass()
