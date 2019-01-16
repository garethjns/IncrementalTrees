# Incremental trees v0.1

Adds partial fit method to sklearn's forest estimators to allow [incremental training](https://scikit-learn.org/0.15/modules/scaling_strategies.html) without being limited to a linear model. Works with [Dask-ml's Incremental](http://ml.dask.org/incremental.html).

These methods don't try and implement partial fitting for decision trees, rather they remove requirement that individual decision trees within forests are trained with the same data (or equally sized bootstraps).

The resulting forests are not "true" online learners, in that batch size affects performance, but they should have similar performance as their standard versions after seeing a similar number of training rows.

## Installing package


Quick start:

1) Clone repo and build pip installable package.
   ````bash
   git clone https://github. com/garethjns/IncrementalTrees.git
   python -m pip install --upgrade pip setuptools wheel
   cd IncrementalTrees
   python3 setup.py sdist bdist_wheel
   ````
3) pip install
   ````bash
   pip install [the .tar.gz or .whl in dist/]
   ````
# Usage
Currently a Streaming version of RandomForestClassifier (StreamingRFC) is implemented in incremental_trees.trees. This works for binary and multiclass classification.

## Examples
See [notes/PerformanceComparisons.ipynb](https://github.com/garethjns/IncrementalTrees/blob/master/notes/PerformanceComparisons.ipynb) and  [notes/PerformanceComparisonsDask.ipynb](https://github.com/garethjns/IncrementalTrees/blob/master/notes/PerformanceComparisons.ipynb) for more examples and performance comparisons against RandomForest.

### Feeding .partial_fit() manually 

#### Incremental forest
Multiple decision trees per subset, with different feature subsets.

````python
import numpy as np
from sklearn.datasets import make_blobs
from incremental_trees.trees import StreamingRFC

srfc = StreamingRFC(n_estimators_per_chunk=20,
                    max_n_estimators=np.inf,
                    n_jobs=4)

# Generate some data in memory
x, y = make_blobs(n_samples=int(2e5), random_state=0, n_features=40,
                  centers=2, cluster_std=100)

# Feed .partial_fit() with random samples of the data
n_chunks = 30
chunk_size = int(2e3)
for i in range(n_chunks):
    sample_idx = np.random.randint(0, x.shape[0], chunk_size)
    # Call .partial_fit(), specifying expected classes
    srfc.partial_fit(x[sample_idx, :], y[sample_idx],
                     classes=np.unique(y))
           
# Should be n_chunks * n_estimators_per_chunk             
print(len(srfc.estimators_))
print(srfc.score(x, y))
````

#### "Incremental" decision trees
Single (or few) decision trees per data subset, with all features.
````python
import numpy as np
from sklearn.datasets import make_blobs
from incremental_trees.trees import StreamingRFC

# Generate some data in memory
x, y = make_blobs(n_samples=int(2e5), random_state=0, n_features=40,
                  centers=2, cluster_std=100)
                  
srfc = StreamingRFC(n_estimators_per_chunk=1,
                    max_n_estimators=np.inf,
                    max_features=x.shape[1])

# Feed .partial_fit() with random samples of the data
n_chunks = 30
chunk_size = int(2e3)
for i in range(n_chunks):
    sample_idx = np.random.randint(0, x.shape[0], chunk_size)
    # Call .partial_fit(), specifying expected classes
    srfc.partial_fit(x[sample_idx, :], y[sample_idx],
                     classes=np.unique(y))
               
# Should be n_chunks * n_estimators_per_chunk      
print(len(srfc.estimators_))
print(srfc.score(x, y))
````

### Feeding .partial_fit() with Dask
Call .fit() directly.

````python
import numpy as np
import dask_ml.datasets
from dask_ml.wrappers import Incremental
from dask.distributed import Client, LocalCluster
from dask import delayed
from incremental_trees.trees import StreamingRFC

# Generate some data out-of-core
x, y = dask_ml.datasets.make_blobs(n_samples=2e5, chunks=1e4, random_state=0,
                                   n_features=40, centers=2, cluster_std=100)

# Create throwawy cluster and client to run on                                  
with LocalCluster(processes=False, 
                  n_workers=2, 
                  threads_per_worker=2) as cluster, Client(cluster) as client:

    # Wrap model with Dask Incremental
    srfc = Incremental(StreamingRFC(n_estimators_per_chunk=10,
                                    max_n_estimators=np.inf,
                                    n_jobs=4))
    
    # Call fit directly, specifying the expect classes
    srfc.fit(x, y,
             classes=delayed(np.unique(y)).compute())
             
    print(len(srfc.estimators_))
    print(srfc.score(x, y))
````
