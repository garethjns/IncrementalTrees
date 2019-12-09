import unittest

import numpy as np
from dask_ml.wrappers import Incremental

from incremental_trees.trees import StreamingEXTC, StreamingEXTR, StreamingRFC, StreamingRFR
from tests.unit.base import DaskTests


class TestDaskModel_1(DaskTests, unittest.TestCase):
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


class TestDaskModel_2(DaskTests, unittest.TestCase):
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


class TestDaskModel_3(DaskTests, unittest.TestCase):
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


class TestDaskModel_4(DaskTests, unittest.TestCase):
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


class TestDaskModel_5(DaskTests, unittest.TestCase):
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


class TestDaskModel_6(DaskTests, unittest.TestCase):
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


class TestDaskModel_7(DaskTests, unittest.TestCase):
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


class TestDaskModel_8(DaskTests, unittest.TestCase):
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


class TestDaskRFC_1(DaskTests, unittest.TestCase):
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


class TestDaskRFC_2(DaskTests, unittest.TestCase):
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


class TestDaskRFC_3(DaskTests, unittest.TestCase):
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


class TestDaskRFC_4(DaskTests, unittest.TestCase):
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


class TestDaskRFR_1(DaskTests, unittest.TestCase):
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


class TestDaskRFR_2(DaskTests, unittest.TestCase):
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


class TestDaskRFR_3(DaskTests, unittest.TestCase):
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


class TestDaskRFR_4(DaskTests, unittest.TestCase):
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
