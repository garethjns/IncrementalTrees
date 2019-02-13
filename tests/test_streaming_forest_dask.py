import unittest
import numpy as np
from incremental_trees.trees import StreamingRFC, StreamingRFR
from sklearn.datasets import make_blobs
from dask.distributed import Client, LocalCluster
import dask_ml
import dask_ml.datasets
from dask_ml.wrappers import Incremental


class Common:
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
        """
        Prepare dask connection once.
        """

        try:
            cls.cluster = LocalCluster(processes=True,
                                       n_workers=4,
                                       threads_per_worker=2,
                                       scheduler_port=8586,
                                       diagnostics_port=8587)
        except (OSError, AttributeError):
            cls.cluster = 'localhost:8586'

        cls.client = Client(cls.cluster)

        # Set helper valuez
        cls.samples_per_chunk = int(cls.n_samples / cls.n_chunks)

    def _prep_data(self, reg=False):
        self.n_samples = 1e5
        self.chunk_size = 1e4
        self.n_chunks = np.ceil(self.n_samples / self.chunk_size).astype(int)

        if reg:
            self.x, self.y = dask_ml.datasets.make_regression(n_samples=self.n_samples,
                                                              chunks=self.chunk_size,
                                                              random_state=0,
                                                              n_features=40)
        else:
            self.x, self.y = dask_ml.datasets.make_blobs(n_samples=self.n_samples,
                                                         chunks=self.chunk_size,
                                                         random_state=0,
                                                         n_features=40,
                                                         centers=2,
                                                         cluster_std=100)

        return self

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        if type(cls.cluster) != str:
            cls.cluster.close()

    def test_fit(self):
        """Test the supplied model by wrapping with dask Incremental and calling .fit."""
        self.mod.fit(self.x, self.y,
                     classes=np.unique(self.y))

        # Set expected number of estimators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod.estimator.n_estimators_per_chunk * self.n_chunks), self.mod.estimator.max_n_estimators)
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


class TestDaskRFC_1(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingRFC(n_estimators_per_chunk=1,
                                           max_n_estimators=39,
                                           verbose=1))

        # Set expected number of estimators
        # This should be set manually depending on data.
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskRFC_2(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingRFC(n_estimators_per_chunk=2,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 20

        # Set helper values
        super().setUpClass()


class TestDaskRFC_3(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingRFC(n_estimators_per_chunk=20,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 200

        # Set helper values
        super().setUpClass()


class TestDaskRFC_4(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingRFC(n_estimators_per_chunk=1,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           max_features=cls.x.shape[1],
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskRFR_1(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingRFR(n_estimators_per_chunk=1,
                                           max_n_estimators=39,
                                           verbose=1))

        # Set expected number of estimators
        # This should be set manually depending on data.
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskRFR_2(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingRFR(n_estimators_per_chunk=2,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 20

        # Set helper values
        super().setUpClass()


class TestDaskRFR_3(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingRFR(n_estimators_per_chunk=20,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 200

        # Set helper values
        super().setUpClass()


class TestDaskRFR_4(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingRFR(n_estimators_per_chunk=1,
                                           n_jobs=-1,
                                           max_n_estimators=np.inf,
                                           max_features=cls.x.shape[1],
                                           verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()
