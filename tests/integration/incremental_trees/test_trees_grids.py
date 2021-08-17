# TODO: These tests aren't finished. Need to generalise, add EXTC, regressors, etc.

import unittest
import warnings

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.trees import StreamingRFC
from tests.common.data_fixture import DataFixture
from tests.common.param_fixtures import RFCGRID, SRFCGRID


class GridBenchmarks:
    def test_fit_all(self):
        """
        Fit grids and compare.

        TODO: Generalise naming.
        """

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            self.rfc_grid.fit(self.x_train, self.y_train)
            self.srfc_grid.fit(self.x_train, self.y_train)

        self.rfc_report, self.rfc_train_auc, self.rfc_test_auc = self._mod_report(mod=self.rfc_grid.best_estimator_)
        self.srfc_report, self.srfc_train_auc, self.srfc_test_auc = self._mod_report(mod=self.srfc_grid.best_estimator_)

        print("==Not-necessarily fair grid comparison==")
        print(f"self.rfc grid score test AUC: {self.rfc_test_auc}")
        print(f"self.srfc grid score test AUC: {self.srfc_test_auc}")
        print(self.srfc_grid.get_params())


class RFCBenchmarkGrid(GridBenchmarks, DataFixture, unittest.TestCase):
    """
    Check a grid runs, assert performance (not added yet)
    """

    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls)

        n_iter = 3
        cls.srfc_grid = RandomizedSearchCV(StreamingRFC(n_jobs=2,
                                                        verbose=1),
                                           param_distributions=SRFCGRID,
                                           scoring='roc_auc',
                                           n_iter=n_iter * 2,
                                           verbose=2,
                                           n_jobs=3,
                                           cv=4)

        cls.rfc_grid = RandomizedSearchCV(RandomForestClassifier(n_jobs=2),
                                          param_distributions=RFCGRID,
                                          scoring='roc_auc',
                                          n_iter=n_iter,
                                          verbose=2,
                                          n_jobs=3,
                                          cv=4)


class EXTCBenchmarkGrid(GridBenchmarks, DataFixture, unittest.TestCase):
    """
    Check a grid runs, assert performance (not added yet)
    """

    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls)

        n_iter = 2
        cls.srfc_grid = RandomizedSearchCV(StreamingEXTC(n_jobs=2,
                                                         verbose=1),
                                           param_distributions=SRFCGRID,
                                           scoring='roc_auc',
                                           n_iter=n_iter * 10,
                                           verbose=2,
                                           n_jobs=3,
                                           cv=4)

        cls.rfc_grid = RandomizedSearchCV(ExtraTreesClassifier(n_jobs=2),
                                          param_distributions=RFCGRID,
                                          scoring='roc_auc',
                                          n_iter=n_iter,
                                          verbose=2,
                                          n_jobs=3,
                                          cv=4)
