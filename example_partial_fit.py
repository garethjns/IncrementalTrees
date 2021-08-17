import numpy as np
from sklearn.datasets import make_blobs

from incremental_trees.models.classification.streaming_rfc import StreamingRFC

if __name__ == "__main__":
    srfc = StreamingRFC(n_estimators_per_chunk=20,
                        max_n_estimators=np.inf,
                        n_jobs=8)

    # Generate some data in memory
    x, y = make_blobs(n_samples=int(2e5), random_state=0, n_features=40,
                      centers=2, cluster_std=100)

    # Feed .partial_fit() with random samples of the data
    n_chunks = 30
    chunk_size = int(2e3)
    for i in range(n_chunks):
        sample_idx = np.random.randint(0, x.shape[0], chunk_size)
        # Call .partial_fit():
        srfc.partial_fit(
            x[sample_idx, :], y[sample_idx],
            classes=np.unique(y),  # Specify expected classes as they many not all be present in this sample
            sample_weight=np.ones_like(sample_idx)  # Optional instance weights
        )

    # Should be n_chunks * n_estimators_per_chunk
    print(len(srfc.estimators_))
    print(srfc.score(x, y))
