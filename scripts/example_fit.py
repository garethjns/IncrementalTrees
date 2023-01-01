import numpy as np
from sklearn.datasets import make_blobs

from incremental_trees.models.classification.streaming_rfc import StreamingRFC

if __name__ == "__main__":
    # Generate some data in memory
    x, y = make_blobs(n_samples=int(2e5), random_state=0, n_features=40, centers=2, cluster_std=100)

    srfc = StreamingRFC(
        n_estimators_per_chunk=3,
        max_n_estimators=np.inf,
        spf_n_fits=30,  # Number of calls to .partial_fit()
        spf_sample_prop=0.3  # Number of rows to sample each on .partial_fit()
    )

    srfc.fit(x, y, sample_weight=np.ones_like(y))  # Optional

    # Should be n_estimators_per_chunk * spf_n_fits
    print(len(srfc.estimators_))
    print(srfc.score(x, y))
