# TODO: These tests aren't finished. Need to generalise, add EXTC, regressors, etc.

import unittest
import warnings
import math
import numpy as np
from incremental_trees.trees import StreamingRFC, StreamingEXTC
from sklearn.ensemble.forest import RandomForestClassifier, ExtraTreesClassifier
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.classification import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from dask.distributed import Client, LocalCluster


class Data:
    def _prep_data(self):
        x, y = make_blobs(n_samples=int(2e4),
                          random_state=0,
                          n_features=40,
                          centers=2,
                          cluster_std=60)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y,
                                                                                test_size=0.25,
                                                                                random_state=123)

        return self

    def _mod_report(self, mod):

        report = classification_report(self.y_test, mod.predict(self.x_test))
        train_auc = roc_auc_score(self.y_train, mod.predict_proba(self.x_train)[:, 1])
        test_auc = roc_auc_score(self.y_test, mod.predict_proba(self.x_test)[:, 1])

        return report, train_auc, test_auc


class PerformanceComparisons(Data):
    """
    Compare srfc to benchmark rfc and logistic regression.

    TODO: Generalise naming, report functions.
    TODO: Set sensible parameters and add performance assets in child tests.
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
        self.rfc_once.fit(self.x_train, self.y_train)

        self.log_reg_report, self.log_reg_train_auc, self.log_reg_test_auc = self._mod_report(self,
                                                                                              mod=self.log_reg)
        self.rfc_report, self.rfc_train_auc, self.rfc_test_auc = self._mod_report(self,
                                                                                  mod=self.rfc)
        self.rfc_once_report, self.rfc_once_train_auc, self.rfc_once_test_auc = self._mod_report(self,
                                                                                                 mod=self.rfc_once)

        return self

    def _assert_same_n_rows(self):
        """
        Assert that given the training settings, the rfc and srfc will, overall, see the same number of rows of data.
        This is a more direct comparison than len(mod.estimators_) as the individual estimators see much less data in
        the srfc case.
        :return:
        """
        # Will be available in actual test.
        n_rows = self.x_train.shape[0]

        self.assertEqual(self.rfc_n_estimators * n_rows,
                         (self.srfc_n_estimators_per_chunk *
                          self.srfc_n_partial_fit_calls *
                          int(n_rows / self.srfc_n_partial_fit_calls)))

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

    def _fit_with_dask(self):

        with LocalCluster(processes=False,
                          n_workers=2,
                          threads_per_worker=2,
                          scheduler_port=8080,
                          diagnostics_port=8081) as cluster, Client(cluster) as client:
            self.srfc_dask.fit(self.x_train, self.y_train)

    def _fit_with_spf(self):

        self.srfc_spf.fit(self.x_train, self.y_train)

    def test_benchmark_manual_random(self):
        """
        Compare models where srfc is trained on a number of manual-random samples from the training data.
        """

        self.srfc_sam = self._fit_srfc(sequential=False)
        self.srfc_sam_report, self.srfc_sam_train_auc, self.srfc_sam_test_auc = self._mod_report(mod=self.srfc_sam)

        print("==Manual feeding partial_fit with random samples==")
        print(f"self.log_reg score test AUC: {self.log_reg_test_auc}")
        print(f"self.rfc score test AUC: {self.rfc_test_auc}")
        print(f"self.srfc_sam score test AUC: {self.srfc_sam_test_auc}")

        # self.assertTrue(math.isclose(self.srfc_sam_test_auc, self.rfc_test_auc,
        #                              rel_tol=0.05))
        # self.assertTrue(math.isclose(self.srfc_sam_test_auc, self.log_reg_test_auc,
        #                              rel_tol=0.05))

        # self._assert_same_n_rows()

    def test_benchmark_auto_spf(self):
        self._fit_with_spf()
        self.srfc_spf_report, self.srfc_spf_train_auc, self.srfc_spf_test_auc = self._mod_report(mod=self.srfc_spf)

        print("==Auto feeding partial_fit with spf samples==")
        print(f"self.log_reg score test AUC: {self.log_reg_test_auc}")
        print(f"self.rfc score test AUC: {self.rfc_test_auc}")
        print(f"self.srfc_spf score test AUC: {self.srfc_spf_test_auc}")

    def test_benchmark_auto_dask(self):
        self._fit_with_dask()
        self.srfc_dask_report, self.srfc_dask_train_auc, self.srfc_dask_test_auc = \
        self._mod_report(mod=self.srfc_dask)

        print("==Auto feeding partial_fit with dask==")
        print(f"self.log_reg score test AUC: {self.log_reg_test_auc}")
        print(f"self.rfc_once score test AUC: {self.rfc_once_test_auc}")
        print(f"self.srfc_dask score test AUC: {self.srfc_dask_test_auc}")

    def test_benchmark_manual_sequential(self):
        """
        Compare models where srfc is trained on sequential chunks of the data once.
        """
        self.srfc_seq = self._fit_srfc(sequential=True)
        self.srfc_seq_report, self.srfc_seq_train_auc, self.srfc_seq_test_auc = self._mod_report(mod=self.srfc_seq)

        print("==Manual feeding partial_fit with sequential samples==")
        print(f"self.log_reg score test AUC: {self.log_reg_test_auc}")
        print(f"self.rfc_once score test AUC: {self.rfc_once_test_auc}")
        print(f"self.srfc score test AUC: {self.srfc_seq_test_auc}")

        # self.assertTrue(math.isclose(self.srfc_seq_test_auc, self.rfc_test_auc,
        #                              rel_tol=0.05))
        # self.assertTrue(math.isclose(self.srfc_seq_test_auc, self.log_reg_test_auc,
        #                              rel_tol=0.05))

        # self._assert_same_n_rows()

    def _generate_comparable_models(self,
                                    srfc_n_estimators_per_chunk: int,
                                    srfc_n_partial_fit_calls: int,
                                    srfc_sample_prop: float,
                                    n_jobs: int=4):
        """
        Set values for streaming models and different set ups. Create two comparable rfcs designed to see
        equivalent numbers of rows.

        Two RFCs are required to compare to different settings. One should see the equivalent of all the data once,
        the other should see more. This should cover the following srfc model combinations:

        - "Manual" feeding (using .partial_fit):
          - Sequential: Will see all the data once (sample size is n / n_partial_fit_calls) per estimator
          - Random: Will see sample_prop * n * n_partial_fit per estimator
        - "Auto" feeding
          - spf: (dask_feeding==False). Will see Will see sample_prop * n * n_partial_fit per estimator.
          - dask: (dask_feeding=True). Will see all the data once (sample size is n / n_partial_fit_calls)
                   per estimator (?) Need to verify this.

        :param srfc_n_estimators_per_chunk: Number of estimators per chunk.
        :param srfc_n_partial_fit_calls: Number of calls to partial fit. Either used manual-sequential or manual-random,
                                         or supplied to fit to handle. In the case of manual-sequential, the size of the
                                         sample is set dynamically to split the data into this number of chunks (all
                                         data is seen once).
        :param srfc_sample_prop: The proportion of data to sample in when feeding .partial_fit with manual-random or
                                 by using .fit.
        :return:
        """
        self.srfc_n_estimators_per_chunk = srfc_n_estimators_per_chunk
        self.srfc_n_partial_fit_calls = srfc_n_partial_fit_calls
        self.srfc_sample_prop = srfc_sample_prop

        # Number of estimators for RFC
        # Set so overall it will see an equal number of rows to the srfc using spf
        self.rfc_n_estimators = int(self.srfc_n_estimators_per_chunk * self.srfc_n_partial_fit_calls
                                    * self.srfc_sample_prop)

        # Make another that will see the same number of rows as the models that see the data once
        self.rfc_once_n_estimators = int(self.srfc_n_estimators_per_chunk * self.srfc_n_partial_fit_calls)

        self.rfc = RandomForestClassifier(n_estimators=self.rfc_n_estimators,
                                          n_jobs=n_jobs)

        self.rfc_once = RandomForestClassifier(n_estimators=self.rfc_once_n_estimators,
                                               n_jobs=n_jobs)

        # "Manual-sequential" and "manual-random" srfc
        # Parameters are the same and object is cloned before fitting.
        self.srfc = StreamingRFC(n_estimators_per_chunk=self.srfc_n_estimators_per_chunk,
                                 n_jobs=n_jobs)

        # "Auto-spf" srfc
        self.srfc_spf = StreamingRFC(dask_feeding=False,
                                     n_estimators_per_chunk=self.srfc_n_estimators_per_chunk,\
                                     spf_n_fits=self.srfc_n_partial_fit_calls,
                                     spf_sample_prop=self.srfc_sample_prop,
                                     n_jobs=n_jobs)
        # "Auto-dask" srfc
        self.srfc_dask = StreamingRFC(dask_feeding=True,
                                      n_estimators_per_chunk=self.srfc_n_estimators_per_chunk,
                                      n_jobs=n_jobs)


class RFCBenchmark1(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=2,
                                        srfc_n_partial_fit_calls=20,
                                        srfc_sample_prop=0.2)

        cls._fit_benchmarks(cls)


class RFCBenchmark2(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=10,
                                        srfc_n_partial_fit_calls=10,
                                        srfc_sample_prop=0.2)

        cls._fit_benchmarks(cls)


class RFCBenchmark3(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=1,
                                        srfc_n_partial_fit_calls=10,
                                        srfc_sample_prop=0.3)

        cls._fit_benchmarks(cls)


class ExtBenchmark1(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=2,
                                        srfc_n_partial_fit_calls=20,
                                        srfc_sample_prop=0.2)

        cls._fit_benchmarks(cls)


class ExtBenchmark2(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=10,
                                        srfc_n_partial_fit_calls=10,
                                        srfc_sample_prop=0.2)

        cls._fit_benchmarks(cls)


class ExtBenchmark3(PerformanceComparisons, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._generate_comparable_models(cls,
                                        srfc_n_estimators_per_chunk=1,
                                        srfc_n_partial_fit_calls=10,
                                        srfc_sample_prop=0.3)

        cls._fit_benchmarks(cls)


RFCGRID = {'min_samples_leaf': [8, 16, 32, 64],
           'min_samples_split': [8, 16, 32, 64],
           'max_features': ['log2', 'sqrt', 0.1, 0.2]}
SRFCGRID = RFCGRID.copy()
SRFCGRID.update({'srfc_n_estimators_per_chunk': [1, 2, 4, 8, 12, 16, 20],
                 'dask_feeding': [False],
                 'spf_n_fits': [10, 20, 30, 40],
                 'spf_sample_prop': [0.1, 0.2, 0.3, 0.4]})
RFCGRID.update({'n_estimators': [100]})


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


class RFCBenchmarkGrid(GridBenchmarks, Data, unittest.TestCase):
    """
    Check a grid runs, assert performance (not added yet)
    """

    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls)

        n_iter = 10
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


class EXTCBenchmarkGrid(GridBenchmarks, Data, unittest.TestCase):
    """
    Check a grid runs, assert performance (not added yet)
    """

    @classmethod
    def setUpClass(cls):
        cls._prep_data(cls)

        n_iter = 20
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
