import unittest
import numpy as np
from incremental_trees.trees import StreamingRFC
from sklearn.datasets import make_blobs
from dask.distributed import Client, LocalCluster
import dask_ml
import dask_ml.datasets
from dask_ml.wrappers import Incremental


class TestBasic(unittest.TestCase):
    """
    Run fits for different model settings, lengths of data, and chunk sizes.
    Checks number of esitmators fit is as expected.

    These are run without using Dask, so the subset passing to partial_fit is handled manually.
    """
    def setUp(self) -> None:
        self.n_samples_1 = 1000
        self.n_samples_2 = 3000

        self.mod_1 = StreamingRFC(n_estimators=1,
                                  max_n_estimators=20)
        self.mod_2 = StreamingRFC(n_estimators=1,
                                  max_n_estimators=np.inf)
        self.mod_3 = StreamingRFC(n_estimators=3,
                                  max_n_estimators=20)
        self.mod_4 = StreamingRFC(n_estimators=3,
                                  max_n_estimators=np.inf)
        self.x_1, self.y_1 = make_blobs(n_samples=self.n_samples_1,
                                        centers=2,
                                        n_features=40,
                                        random_state=0)
        self.x_2, self.y_2 = make_blobs(n_samples=self.n_samples_2,
                                        centers=3,
                                        n_features=26,
                                        random_state=1)

    def test_run_fits_data1_chunks10(self) -> None:
        """
        Run fits on all models using self.x_1 and self.y_1. Use a chunk size of 10.
        """
        chunk_size = 10
        n_chunks = int(self.n_samples_1 / chunk_size)
        samples_per_chunk = int(self.n_samples_1 / n_chunks)

        s_idx = 0
        e_idx = samples_per_chunk

        # Call the first partial fit specifying classes
        self.mod_1.partial_fit(self.x_1[s_idx:e_idx, :],
                               self.y_1[s_idx:e_idx],
                               classes=np.unique(self.y_1))

        self.mod_2.partial_fit(self.x_1[s_idx:e_idx, :],
                               self.y_1[s_idx:e_idx],
                               classes=np.unique(self.y_1))

        self.mod_3.partial_fit(self.x_1[s_idx:e_idx, :],
                               self.y_1[s_idx:e_idx],
                               classes=np.unique(self.y_1))

        self.mod_4.partial_fit(self.x_1[s_idx:e_idx, :],
                               self.y_1[s_idx:e_idx],
                               classes=np.unique(self.y_1))

        # TODO: Some asserts here or split test?

        # Call the rest
        for i in range(1, n_chunks):
            print(s_idx)
            print(e_idx)
            self.mod_1.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

            self.mod_2.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

            # Leave the classes kwarg in a couple, should be ignored.
            self.mod_3.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx],
                                   classes=np.unique(self.y_1))

            self.mod_4.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx],
                                   classes=np.unique(self.y_1))

            s_idx = e_idx
            e_idx = s_idx + samples_per_chunk

        # Should be min of ests per chunk * n_chunks, or max_n_esitmators.
        expect_1 = min((self.mod_1._estimators_per_chunk * n_chunks), self.mod_1.max_n_estimators)
        self.assertEqual(expect_1, 20)
        self.assertEqual(len(self.mod_1.estimators_), expect_1)

        expect_2 = min((self.mod_2._estimators_per_chunk * n_chunks), self.mod_2.max_n_estimators)
        self.assertEqual(expect_2, 100)
        self.assertEqual(len(self.mod_2.estimators_), expect_2)

        # First multiple of 3 above 20?
        expect_3 = np.arange(0, n_chunks, 1) * self.mod_3._estimators_per_chunk
        expect_3 = np.min(expect_3[expect_3 > self.mod_3.max_n_estimators])
        self.assertEqual(expect_3, 21)
        self.assertEqual(len(self.mod_3.estimators_), expect_3)

        expect_4 = min((self.mod_4._estimators_per_chunk * n_chunks), self.mod_4.max_n_estimators)
        self.assertEqual(expect_4, 300)
        self.assertEqual(len(self.mod_4.estimators_), expect_4)


class TestBasicDask(unittest.TestCase):
    """
    Run fits for different model settings, lengths of data, and chunk sizes.
    Checks number of estimators fit is as expected.
    """
    @classmethod
    def setUpClass(cls) -> None:
        """
        Prepare dask connection once.
        """
        # super().setUpClass()

        try:
            cls.cluster = LocalCluster(processes=True,
                                       n_workers=4,
                                       threads_per_worker=2,
                                       scheduler_port=8585,
                                       diagnostics_port=8586)
        except RuntimeError:
            cls.cluster = 'localhost:8585'

        cls.client = Client(cls.cluster)

        cls.x, cls.y = dask_ml.datasets.make_blobs(n_samples=2e5,
                                                   chunks=1e4,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.n_chunks = len(cls.x.chunks[0])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        if type(cls.cluster) != 'str':
            cls.cluster.close()

    def test_fit_incremental_forest(self) -> None:
        n_estimators_per_chunk = 1
        srfc = Incremental(StreamingRFC(n_estimators=n_estimators_per_chunk,
                                        n_jobs=-1,
                                        max_n_estimators=np.inf))

        srfc.fit(self.x, self.y,
                 classes=np.unique(self.y))

        expected_ests = self.n_chunks * n_estimators_per_chunk
        self.assertAlmostEquals(len(srfc.estimator_), expected_ests)

    def test_fit_incremental_forest_multiple_ests_per_chunk(self) -> None:
        n_estimators_per_chunk = 20
        srfc = Incremental(StreamingRFC(n_estimators=20,
                                        n_jobs=-1,
                                        max_n_estimators=np.inf))

        srfc.fit(self.x, self.y,
                 classes=np.unique(self.y))

        expected_ests = self.n_chunks * n_estimators_per_chunk
        self.assertAlmostEquals(len(srfc.estimator_), expected_ests)

    def test_fit_partial_dtc(self) -> None:
        n_estimators_per_chunk = 10
        srfc = Incremental(StreamingRFC(n_estimators=n_estimators_per_chunk,
                                        n_jobs=-1,
                                        max_n_estimators=np.inf,
                                        max_features=self.x.shape[1]))

        srfc.fit(self.x, self.y,
                 classes=np.unique(self.y))

        expected_ests = self.n_chunks * n_estimators_per_chunk
        self.assertAlmostEquals(len(srfc.estimator_), expected_ests)

