import setuptools

# with open("README.md", "r") as fh:
#     long_description = fh.read()

setuptools.setup(name='IncrementalTrees',
                 version="0.0.1",
                 author="Gareth Jones",
                 author_email="author@example.com",
                 description='Sklearn forests with partial fits',
                 # long_description=long_description,
                 # long_description_content_type="text/markdown",
                 url="",
                 requires=['dask', 'dask_ml', 'sklearn', 'bokeh'])
