import unittest
import numpy as np
from incremental_trees.trees import StreamingEXTC, StreamingEXTR
from dask_ml.wrappers import Incremental
from tests.test_streaming_forest_dask import Common


class TestDaskModel_1(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=1,
                                            max_n_estimators=39,
                                            verbose=1))

        # Set expected number of estimators
        # This should be set manually depending on data.
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskModel_2(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=2,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 20

        # Set helper values
        super().setUpClass()


class TestDaskModel_3(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=20,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 200

        # Set helper values
        super().setUpClass()


class TestDaskModel_4(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls)
        cls.mod = Incremental(StreamingEXTC(n_estimators_per_chunk=1,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            max_features=cls.x.shape[1],
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskModel_5(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingEXTR(n_estimators_per_chunk=1,
                                            max_n_estimators=39,
                                            verbose=1))

        # Set expected number of estimators
        # This should be set manually depending on data.
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()


class TestDaskModel_6(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingEXTR(n_estimators_per_chunk=2,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 20

        # Set helper values
        super().setUpClass()


class TestDaskModel_7(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingEXTR(n_estimators_per_chunk=20,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 200

        # Set helper values
        super().setUpClass()


class TestDaskModel_8(Common, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingEXTR(n_estimators_per_chunk=1,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            max_features=cls.x.shape[1],
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 10

        # Set helper values
        super().setUpClass()
