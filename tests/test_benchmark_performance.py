import unittest
import math
import numpy as np
from incremental_trees.trees import StreamingRFC
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.base import clone


class PerformanceComparisons:
    """
    Compare srfc to benchmark rfc and logistic regression.
    """
    @classmethod
    def setUpClass(cls):
        """Prepare comparable models"""

        cls.rfc_n_estimators = None
        cls.srfc_n_estimators_per_chunk = None
        cls.srfc_n_partial_fit_calls = None

        cls = cls._prep_data(cls)

        # In both cases just using default params, probably not optimal.
        # TODO: Make a more fair comparison with better parameters?
        cls.log_reg = LogisticRegression()

    def _fit_benchmarks(self):
        """
        Fit log reg and rfc benchmark models.
        :return:
        """

        self.log_reg.fit(self.x_train, self.y_train)
        self.rfc.fit(self.x_train, self.y_train)

        self.log_reg_report, self.log_reg_train_auc, self.log_reg_test_auc = self._mod_report(self,
                                                                                              mod=self.log_reg)
        self.rfc_report, self.rfc_train_auc, self.rfc_test_auc = self._mod_report(self,
                                                                                  mod=self.rfc)

        return self

    def _mod_report(self, mod):

        report = classification_report(self.y_test, mod.predict(self.x_test))
        train_auc = roc_auc_score(self.y_train, mod.predict_proba(self.x_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, mod.predict_proba(self.x_test)[:, 1])

        return report, train_auc, test_auc

    def _prep_data(self):
        x, y = load_breast_cancer(return_X_y=True)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=0.25,
                                                                                random_state=123)

        return self

    def _assert_same_n_rows(self):
        """
        Assert that given the training settings, the rfc and srfc will, overall, see the same number of rows of data.
        This is a more direct comparison than len(mod.estimators_) as the individual estimators see much less data in
        the srfc case.
        :return:
        """
        # Will be available in actual test.
        self.assertEqual(self.rfc_n_estimators, self.srfc_n_estimators_per_chunk * self.srfc_n_partial_fit_calls)

    def _fit_srfc(self,
                  sequential: bool=True,
                  n_prop: float=0.1) -> StreamingRFC:
        """
        Fit the streaming RFC. Total number of rows used in training varies depending on sequential.

        sequential==True
        In this case, rows used per estimator scales with n_estimators. So in total 100% of rows are used for
        training once.
        If there are 10 calls, 10% of data is used in each .partial_fit call. Equivalent n rows to 1 tree.
        If 100 calls, 1% of data used in each .partial_fit call. Still equivalent n rows to 1 tree.
        This is similar to the Dask use case.

        sequential==False
        Randomly sample % of data with replacement n times.
        Set % to sample and n calls, allows over sampling to compare more directly with RandomForest.
        If there are 10 calls and 10% of data is used in each .partial_fit call: Equivalent n rows to 1 tree.
        If 100 calls, 10% of data used in each .partial_fit call: 1000% of rows used, equivalent n rows to 10 trees.

        :param sequential: If true step through all data once. If False, draw n_prop proportions of data n_draws times.
        :param n_prop: When sequential is False, use to set prop of data to draw in each .partial_fit call.

        :return:
        """

        # Clone the parameters from model specified in
        srfc = clone(self.srfc)

        n_rows = self.x_train.shape[0]

        if sequential:
            # Step through all data once
            n_sample_rows = int(n_rows / self.srfc_n_partial_fit_calls)
            sidx = 0
            eidx = n_sample_rows
            for i in range(self.srfc_n_partial_fit_calls):
                idx = np.arange(sidx, eidx)
                srfc.partial_fit(self.x_train[idx, :], self.y_train[idx],
                                 classes=[0, 1])
                sidx = eidx
                eidx = min(eidx + n_sample_rows, n_rows)
        else:
            # Sample n_prop of data self.srfc_n_partial_fit_calls times
            n_sample_rows = int(n_rows * n_prop)
            for i in range(self.srfc_n_partial_fit_calls):
                # Sample indexes with replacement
                idx = np.random.randint(0, n_rows, n_sample_rows)
                srfc.partial_fit(self.x_train[idx, :], self.y_train[idx],
                                 classes=[0, 1])

        return srfc

    def test_benchmark_sample(self):
        """
        Compare models where srfc is train on a number of random samples from the training data.
        """

        self.srfc_sam = self._fit_srfc(sequential=False)
        self.srfc_sam_report, self.srfc_sam_train_auc, self.srfc_sam_test_auc = self._mod_report(mod=self.srfc_sam)

        self.assertTrue(math.isclose(self.srfc_sam_test_auc, self.rfc_test_auc,
                                     rel_tol=0.04))
        self.assertTrue(math.isclose(self.srfc_sam_test_auc, self.log_reg_test_auc,
                                     rel_tol=0.05))

    def test_benchmark_sequential(self):
        """
        Compare models where srfc is train on sequential chunks of the data.
        """
        self.srfc_seq = self._fit_srfc(sequential=True)
        self.srfc_seq_report, self.srfc_seq_train_auc, self.srfc_seq_test_auc = self._mod_report(mod=self.srfc_seq)

        self.assertTrue(math.isclose(self.srfc_seq_test_auc, self.rfc_test_auc,
                                     rel_tol=0.04))
        self.assertTrue(math.isclose(self.srfc_seq_test_auc, self.log_reg_test_auc,
                                     rel_tol=0.05))


class Benchmark1(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.rfc_n_estimators = 10
        cls.srfc_n_estimators_per_chunk = 1
        cls.srfc_n_partial_fit_calls = 10

        cls.rfc = RandomForestClassifier(n_estimators=cls.rfc_n_estimators)
        cls.srfc = StreamingRFC(n_estimators_per_chunk=cls.srfc_n_estimators_per_chunk)

        cls._fit_benchmarks(cls)

        # cls._assert_same_n_rows(cls)


class Benchmark2(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.rfc_n_estimators = 100
        cls.srfc_n_estimators_per_chunk = 10
        cls.srfc_n_partial_fit_calls = 10

        cls.rfc = RandomForestClassifier(n_estimators=cls.rfc_n_estimators)
        cls.srfc = StreamingRFC(n_estimators_per_chunk=cls.srfc_n_estimators_per_chunk)

        cls._fit_benchmarks(cls)

        # cls._assert_same_n_rows(cls)


class Benchmark3(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.rfc_n_estimators = 100
        cls.srfc_n_estimators_per_chunk = 1
        cls.srfc_n_partial_fit_calls = 10

        cls.rfc = RandomForestClassifier(n_estimators=cls.rfc_n_estimators)
        cls.srfc = StreamingRFC(n_estimators_per_chunk=cls.srfc_n_estimators_per_chunk)

        cls._fit_benchmarks(cls)

        # cls._assert_same_n_rows(cls)