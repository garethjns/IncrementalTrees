import numpy as np
import sklearn
from dask_ml.datasets import make_blobs, make_regression
from distributed import LocalCluster, Client
from sklearn import clone
from sklearn.model_selection import RandomizedSearchCV


class PredictTests:
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
        self.mod.score(self.x, self.y)


class PartialFitTests(PredictTests):
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

        # Set expected number of esitmators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod.n_estimators_per_chunk * self.n_chunks), self.mod.max_n_estimators)
        self.assertEqual(expect_, self.expected_n_estimators)
        # Then check the model matches the validated expectation
        self.assertEqual(len(self.mod.estimators_), self.expected_n_estimators)

    def test_result(self):
        """Test performance of model is approximately as expected."""
        pass


class FitTests(PredictTests):
    """
    Test direct calls to.fit with dask off, which will use ._sampled_partial_fit() to feed partial_fit.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set helper actual model from specified values. Need to super this from child setUpClass.
        :return:
        """
        cls.expected_n_estimators = cls.spf_n_fits * cls.n_estimators_per_sample

        cls.n_samples = 1000
        cls.x, cls.y = sklearn.datasets.make_blobs(n_samples=cls.n_samples,
                                                   random_state=0,
                                                   n_features=40,
                                                   centers=2,
                                                   cluster_std=100)

        cls.grid = RandomizedSearchCV(clone(cls.mod),
                                      scoring='roc_auc',
                                      cv=2,
                                      n_iter=3,
                                      verbose=10,
                                      param_distributions={'spf_sample_prop': [0.1, 0.2, 0.3],
                                                           'spf_n_fits': [10, 20, 30]},
                                      n_jobs=-1)

    def test_fit__sampled_partial_fit(self):
        """With dask off, call .fit directly."""
        self.mod.fit(self.x, self.y)

    def test_n_estimators(self):
        self.assertEqual(self.expected_n_estimators, len(self.mod.estimators_))

    def test_grid_search(self):
        """With dask off, try with sklearn GS."""
        self.grid.fit(self.x, self.y)
        self.grid.score(self.x, self.y)


class DaskTests:
    """
    Standard tests to run on supplied model and data.

    Inherit this into a class with model/data defined in setUpClass into self.mod, self.x, self.y. Then call the
    setupClass method here to set some helper values.

    These tests need to run in order, as self.mod used through tests. Maybe would be better to mock it each time,
    but lazy....

    These are run without using Dask, so the subset passing to partial_fit is handled manually.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set helper values from specified model/data. Need to super this from child setUpClass.
        :return:
        """
        """
        Prepare dask connection once.
        """

        try:
            cls.cluster = LocalCluster(processes=True,
                                       n_workers=4,
                                       threads_per_worker=2,
                                       scheduler_port=8586,
                                       diagnostics_port=8587)
        except (OSError, AttributeError):
            cls.cluster = 'localhost:8586'

        cls.client = Client(cls.cluster)

        # Set helper valuez
        cls.samples_per_chunk = int(cls.n_samples / cls.n_chunks)

    def _prep_data(self, reg=False):
        self.n_samples = int(1e5)
        self.chunk_size = int(1e4)
        self.n_chunks = np.ceil(self.n_samples / self.chunk_size).astype(int)

        if reg:
            self.x, self.y = make_regression(n_samples=self.n_samples,
                                             chunks=self.chunk_size,
                                             random_state=0,
                                             n_features=40)
        else:
            self.x, self.y = make_blobs(n_samples=self.n_samples,
                                        chunks=self.chunk_size,
                                        random_state=0,
                                        n_features=40,
                                        centers=2,
                                        cluster_std=100)

        return self

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
        if type(cls.cluster) != str:
            cls.cluster.close()

    def test_fit(self):
        """Test the supplied model by wrapping with dask Incremental and calling .fit."""
        self.mod.fit(self.x, self.y,
                     classes=np.unique(self.y).compute())

        # Set expected number of estimators in class set up
        # Check it matches with parameters
        expect_ = min((self.mod.estimator.n_estimators_per_chunk * self.n_chunks), self.mod.estimator.max_n_estimators)
        self.assertEqual(expect_, self.expected_n_estimators)
        # Then check the model matches the validated expectation
        self.assertEqual(len(self.mod.estimators_), self.expected_n_estimators)

    def test_predict(self):
        """
        Test prediction function runs are returns expected shape, even if all classes are not in prediction set.
        :return:
        """

        # Predict on all data
        self.mod.predict(self.x)

        # Predict on single row
        self.mod.predict(self.x[0, :].reshape(1, -1))

    def test_result(self):
        """Test performance of model is approximately as expected."""
        pass
