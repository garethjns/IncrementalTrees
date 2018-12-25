import unittest
import numpy as np
from incremental_trees.trees import StreamingRFC
from sklearn.datasets import make_blobs


class TestBasic(unittest.TestCase):
    """
    Run fits for different model settings, lengths of data, and chunk sizes.
    Checks number of esitmators fit is as expected.

    These are run without using Dask, so the subset passing to partial_fit is handled manually.
    """
    def setUp(self):
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
        for i in range(0, n_chunks):
            print(s_idx)
            print(e_idx)
            self.mod_1.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

            self.mod_2.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

            self.mod_3.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

            self.mod_4.partial_fit(self.x_1[s_idx:e_idx, :],
                                   self.y_1[s_idx:e_idx])

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
    Checks number of esitmators fit is as expected.
    """
    def setUpClass(cls):
        """
        Prepare dask connection
        :return:
        """
        pass
