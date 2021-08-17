# Incremental trees
![The overcomplicated tests are...](https://github.com/garethjns/IncrementalTrees/workflows/The%20overcomplicated%20tests%20are.../badge.svg)

Adds partial fit method to sklearn's forest estimators (currently RandomForestClassifier/Regressor and ExtraTreesClassifier/Regressor) to allow [incremental training](https://scikit-learn.org/0.15/modules/scaling_strategies.html) without being limited to a linear model. Works with or without [Dask-ml's Incremental](http://ml.dask.org/incremental.html).

These methods don't try and implement partial fitting for decision trees, rather they remove requirement that individual decision trees within forests are trained with the same data (or equally sized bootstraps). This reduces memory burden, training time, and variance. This is at the cost of generally increasing the number of weak learners will probably be required. 

The resulting forests are not "true" online learners, as batch size affects performance. However, they should have similar (possibly better) performance as their standard versions after seeing an equivalent number of training rows.

## Installing package

Quick start:

1) Clone repo and build pip installable package.
   ````bash
    pip install incremental_trees
   ````


## Usage Examples
Currently implemented:
 - Streaming versions of RandomForestClassifier (StreamingRFC) and ExtraTreesClassifer (StreamingEXTC). They work should work for binary and multi-class classification, but not multi-output yet.
 - Streaming versions of RandomForestRegressor (StreamingRFR) and ExtraTreesRegressor (StreamingEXTR). 

See:
- Below for example of using different mechanisms to feed .partial_fit() and different parameter set ups.  
- [notes/PerformanceComparisons.ipynb](https://github.com/garethjns/IncrementalTrees/blob/master/notes/PerformanceComparisons.ipynb) and  [notes/PerformanceComparisonsDask.ipynb](https://github.com/garethjns/IncrementalTrees/blob/master/notes/PerformanceComparisonsDask.ipynb) for more examples and performance comparisons against RandomForest. Also there are some (unfinished) performance comparisons in tests/.


### Data feeding mechanisms

#### Fitting with .fit()
Feeds .partial_fit() with randomly samples rows.


````python
import numpy as np
from sklearn.datasets import make_blobs
from incremental_trees.models.classification.streaming_rfc import StreamingRFC

# Generate some data in memory
x, y = make_blobs(n_samples=int(2e5), random_state=0, n_features=40,
                  centers=2, cluster_std=100)

srfc = StreamingRFC(n_estimators_per_chunk=3,
                    max_n_estimators=np.inf,
                    spf_n_fits=30,  # Number of calls to .partial_fit()
                    spf_sample_prop=0.3)  # Number of rows to sample each on .partial_fit()

srfc.fit(x, y, 
         sample_weight=np.ones_like(y))  # Optional, gets sampled along with the data

# Should be n_estimators_per_chunk * spf_n_fits
print(len(srfc.estimators_))
print(srfc.score(x, y))
````

#### Fitting with .fit() and Dask
Call .fit() directly, let dask handle sending data to .partial_fit()

````python
import numpy as np
import dask_ml.datasets
from dask_ml.wrappers import Incremental
from dask.distributed import Client, LocalCluster
from dask import delayed
from incremental_trees.models.classification.streaming_rfc import StreamingRFC

# Generate some data out-of-core
x, y = dask_ml.datasets.make_blobs(n_samples=2e5, chunks=1e4, random_state=0,
                                   n_features=40, centers=2, cluster_std=100)

# Create throwaway cluster and client to run on                                  
with LocalCluster(processes=False, n_workers=2, 
                  threads_per_worker=2) as cluster, Client(cluster) as client:

    # Wrap model with Dask Incremental
    srfc = Incremental(StreamingRFC(dask_feeding=True,  # Turn dask on
                                    n_estimators_per_chunk=10,
                                    max_n_estimators=np.inf,
                                    n_jobs=4))
    
    # Call fit directly, specifying the expected classes
    srfc.fit(x, y,
             classes=delayed(np.unique)(y).compute())
             
    print(len(srfc.estimators_))
    print(srfc.score(x, y))
````

#### Feeding .partial_fit() manually 
.partial_fit can be called directly and fed data manually.

For example, this can be used to feed .partial_fit() sequentially (although below example selects random rows, which is similar to non-dask example above).

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
    # Call .partial_fit(), specifying expected classes, also supports other .fit args such as sample_weight
    srfc.partial_fit(x[sample_idx, :], y[sample_idx],
                     classes=np.unique(y))
           
# Should be n_chunks * n_estimators_per_chunk             
print(len(srfc.estimators_))
print(srfc.score(x, y))
````

### Possible model set ups
There are a couple of different model setups worth considering. No idea which works best. 

#### "Incremental forest"
For the number of chunks/fits, sample rows from X, then fit a number of single trees (with different column subsets), eg.
````python
srfc = StreamingRFC(n_estimators_per_chunk=10,
                    max_features='sqrt')    
````
#### "Incremental decision trees"
Single (or few) decision trees per data subset, with all features. 
````python
srfc = StreamingRFC(n_estimators_per_chunk=1,
                    max_features=x.shape[1])
````

# Version history
## v0.5.1
 - Add support for passing fit args/kwargs via `.fit` (specifically, `sample_weight`)
## v0.5.0
 - Add support for passing fit args/kwargs via `.partial fit` (specifically, `sample_weight`)
## v0.4.0
 - Refactor and tidy, try with new versions of Dask/sklearn
## v0.3.1-3
  - Update Dask versions
## v0.3.0
  - Updated unit tests
  - Added performance benchmark tests for classifiers, not finished.
  - Added regressor versions of RandomForest (StreamingRFR) and ExtaTrees (StreamingEXTR, also renamed StreamingEXT to StreamingEXTC).
  - .fit() overload to handle feeding .partial_fit() with random row samples, without using Dask. Adds compatibility with sklearn SearchCV objects.
## v0.2.0
  - Add ExtraTreesClassifier (StreamingEXT)
## v0.1.0
  - .partial_fit() for RandomForestClassifier (StreamingRFC)
  - .predict_proba() for RandomforestClassifier
  
  
