
RFCGRID = {'min_samples_leaf': [8, 16, 32, 64],
           'min_samples_split': [8, 16, 32, 64],
           'max_features': ['log2', 'sqrt', 0.1, 0.2]}

SRFCGRID = RFCGRID.copy()
SRFCGRID.update({'srfc_n_estimators_per_chunk': [1, 2, 4, 8, 12, 16, 20],
                 'dask_feeding': [False],
                 'spf_n_fits': [10, 20, 30, 40],
                 'spf_sample_prop': [0.1, 0.2, 0.3, 0.4]})
RFCGRID.update({'n_estimators': [100]})
