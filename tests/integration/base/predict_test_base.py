import unittest
from typing import Union

import numpy as np

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR


class PredictTestBase(unittest.TestCase):
    x: np.ndarray
    y: np.ndarray
    mod: Union[StreamingEXTC, StreamingEXTR, StreamingRFC, StreamingRFR]

    def test_predict(self):
        """
        Test prediction function runs are returns expected shape, even if all classes are not in prediction set.
        :return:
        """

        # Predict on all data
        preds = self.mod.predict(self.x)
        self.assertEqual(preds.shape, (self.x.shape[0],))

        # Predict on single row
        preds = self.mod.predict(self.x[0, :].reshape(1, -1))
        self.assertEqual(preds.shape, (1,))

    def test_predict_proba(self):
        """
        Test prediction function runs are returns expected shape, even if all classes are not in prediction set.
        :return:
        """
        if getattr(self.mod, 'predict_proba', False) is False:
            # No predict_proba for this model type
            pass
        else:
            # Predict on all data
            preds = self.mod.predict_proba(self.x)
            self.assertEqual(preds.shape, (self.x.shape[0], 2))

            # Predict on single row
            preds = self.mod.predict_proba(self.x[0, :].reshape(1, -1))
            self.assertEqual(preds.shape, (1, 2))

    def test_score(self):
        score = self.mod.score(self.x, self.y)
        self.assertIsInstance(score, float)
