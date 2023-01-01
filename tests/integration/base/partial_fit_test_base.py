from typing import Union

import numpy as np

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR
from tests.integration.base.predict_test_base import PredictTestBase


class PartialFitTestBase(PredictTestBase):
    mod: Union[StreamingEXTC, StreamingEXTR, StreamingRFC, StreamingRFR]
    x: np.ndarray
    y: np.ndarray
    n_samples: int
    chunk_size: int
    n_chunks: int
    samples_per_chunk: int
    expected_n_estimators: int

    """
    Standard tests to run on supplied model and data.

    Inherit this into a class with model/data defined in setUpClass into self.mod, self.x, self.y. Then call the
    setupClass method here to set some helper values.

    These tests need to run in order, as self.mod used through tests. Maybe would be better to mock it each time,
    but lazy....

    These are run without using Dask, so the subset passing to partial_fit is handled manually.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """
        Set helper values from specified model/data. Need to super this from child setUpClass.
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

        # Set expected number of estimators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod.n_estimators_per_chunk * self.n_chunks), self.mod.max_n_estimators)
        self.assertEqual(expect_, self.expected_n_estimators)
        # Then check the model matches the validated expectation
        self.assertEqual(len(self.mod.estimators_), self.expected_n_estimators)
