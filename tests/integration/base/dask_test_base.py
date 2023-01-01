import unittest
from typing import Union

import numpy as np
from dask_ml.datasets import make_blobs, make_regression
from dask_ml.wrappers import Incremental
from distributed import LocalCluster, Client


class DaskTestBase(unittest.TestCase):
    client: Client
    cluster: Union[str, LocalCluster]
    n_samples: int
    n_chunks: int
    n_samples: int
    mod: Incremental
    reg: bool
    expected_n_estimators: int
    x: np.ndarray
    y: np.ndarray

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
            cls.cluster = LocalCluster(
                processes=True,
                n_workers=4,
                threads_per_worker=2,
                scheduler_port=8586,
                diagnostics_port=8587
            )
        except (OSError, AttributeError):
            cls.cluster = 'localhost:8586'

        cls.client = Client(cls.cluster)
        cls.samples_per_chunk = int(cls.n_samples / cls.n_chunks)

    def _prep_data(self, reg: bool):
        self.n_samples = int(1e5)
        self.chunk_size = int(1e4)
        self.n_chunks = np.ceil(self.n_samples / self.chunk_size).astype(int)

        if reg:
            self.x, self.y = make_regression(
                n_samples=self.n_samples,
                chunks=self.chunk_size,
                random_state=0,
                n_features=40)
        else:
            self.x, self.y = make_blobs(
                n_samples=self.n_samples,
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

    def test_fit_predict(self):
        """Test the supplied model by wrapping with dask Incremental and calling .fit."""

        # Act
        self.mod.fit(self.x, self.y, classes=np.unique(self.y).compute())
        preds = self.mod.predict(self.x)
        single_pred = self.mod.predict(self.x[0, :].reshape(1, -1))

        # Assert
        # Set expected number of estimators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod.estimator.n_estimators_per_chunk * self.n_chunks), self.mod.estimator.max_n_estimators)
        self.assertEqual(expect_, self.expected_n_estimators)
        # Then check the model matches the validated expectation
        self.assertEqual(len(self.mod.estimators_), self.expected_n_estimators)
        self.assertEqual(self.x.shape[0], preds.shape[0])
        self.assertEqual(1, len(single_pred))
