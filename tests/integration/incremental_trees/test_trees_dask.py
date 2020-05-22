import unittest

import numpy as np
from dask_ml.wrappers import Incremental

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from incremental_trees.trees import StreamingRFC
from tests.integration.base import DaskTests


class TestDaskModel1(DaskTests, unittest.TestCase):
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


class TestDaskModel2(DaskTests, unittest.TestCase):
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


class TestDaskModel3(DaskTests, unittest.TestCase):
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


class TestDaskModel4(DaskTests, unittest.TestCase):
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


class TestDaskModel5(DaskTests, unittest.TestCase):
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


class TestDaskModel6(DaskTests, unittest.TestCase):
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


class TestDaskModel7(DaskTests, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up model to test."""
        cls = cls._prep_data(cls,
                             reg=True)
        cls.mod = Incremental(StreamingEXTR(n_estimators_per_chunk=4,
                                            n_jobs=-1,
                                            max_n_estimators=np.inf,
                                            verbose=1))

        # Set expected number of estimators
        cls.expected_n_estimators = 40

        # Set helper values
        super().setUpClass()


class TestDaskModel8(DaskTests, unittest.TestCase):
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


class TestDaskRFC1(DaskTests, unittest.TestCase):
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


class TestDaskRFC2(DaskTests, unittest.TestCase):
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


class TestDaskRFC3(DaskTests, unittest.TestCase):
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


class TestDaskRFC4(DaskTests, unittest.TestCase):
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


class TestDaskRFR1(DaskTests, unittest.TestCase):
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


class TestDaskRFR2(DaskTests, unittest.TestCase):
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


class TestDaskRFR3(DaskTests, unittest.TestCase):
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


class TestDaskRFR4(DaskTests, unittest.TestCase):
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
