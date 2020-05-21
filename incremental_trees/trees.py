import numpy as np

from incremental_trees.models.classification.streaming_extc import StreamingEXTC
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
from incremental_trees.models.regression.streaming_extr import StreamingEXTR
from incremental_trees.models.regression.streaming_rfr import StreamingRFR


def bunch_of_examples():
    from sklearn.datasets import make_blobs, make_regression

    x, y = make_regression(n_samples=int(2e5),
                           random_state=0,
                           n_features=40)

    srfr = StreamingRFR(n_estimators_per_chunk=5,
                        spf_n_fits=10,
                        dask_feeding=False,
                        verbose=0,
                        n_jobs=2)

    srfr.fit(x, y)

    # Fit 10 regressors
    for _ in range(10):
        x, y = make_regression(n_samples=int(2e5),
                               random_state=0,
                               n_features=40)

        srfr = StreamingRFR(n_estimators_per_chunk=5,
                            max_n_estimators=100,
                            verbose=0,
                            n_jobs=5)

        chunk_size = int(2e3)
        for _ in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            srfr.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SRFR: {srfr.score(x, y)}")

        sext = StreamingEXTR(n_estimators_per_chunk=5,
                             max_n_estimators=100,
                             verbose=0,
                             n_jobs=5)

        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            sext.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SEXTR: {sext.score(x, y)}")

    # Fit 10 classifiers
    for _ in range(10):
        x, y = make_blobs(n_samples=int(2e5),
                          random_state=0,
                          n_features=40,
                          centers=2,
                          cluster_std=100)

        srfc = StreamingRFC(n_estimators_per_chunk=5,
                            max_n_estimators=100,
                            verbose=0,
                            n_jobs=5)

        chunk_size = int(2e3)
        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            srfc.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SRFC: {srfc.score(x, y)}")

        sext = StreamingEXTC(n_estimators_per_chunk=5,
                             max_n_estimators=100,
                             verbose=0,
                             n_jobs=5)

        for i in range(20):
            sample_idx = np.random.randint(0, x.shape[0], chunk_size)
            sext.partial_fit(x[sample_idx], y[sample_idx],
                             classes=np.unique(y))

        print(f"SEXTC: {sext.score(x, y)}")


if __name__ == '__main__':
    bunch_of_examples()
