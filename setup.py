import setuptools

from incremental_trees import __version__

setuptools.setup(name='incremental_trees',
                 version=__version__,
                 author="Gareth Jones",
                 author_email="garethgithub@gmail.com",
                 description='Sklearn forests with partial fits',
                 long_description_content_type="text/markdown",
                 packages=setuptools.find_packages(),
                 url="https://github.com/garethjns/IncrementalTrees",
                 install_requires=["scikit-learn>=0.22", "pandas",
                                   "dask>=2",
                                   "dask-glm==0.2.0",
                                   "dask-ml>=1",
                                   "bokeh"])
